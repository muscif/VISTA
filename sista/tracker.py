from collections import defaultdict

import numpy as np
from PIL import Image
from supervision.detection.core import Detections
from trackers import BoTSORTTracker
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils import vocab_mapping


class Embedder:
    def __init__(self, model_name="facebook/dinov2-small"):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
        self.model.eval()

    def embed(self, images: list[Image.Image]) -> list[torch.Tensor]:
        inputs = self.processor(images=images, return_tensors="pt").to("cuda:0")

        with torch.inference_mode():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        return list(embeddings.unbind(0))


class DeepTrackerAM: # Appearance - Motion
    def __init__(self, embedder: Embedder, threshold: float):
        self.embedder = embedder
        self.threshold = threshold
        self.history: dict[int, torch.Tensor] = {}

    def update(self, detections: Detections, frame: Image.Image):
        new_ids = []
        known_ids = []
        crops = []
        query_rows = []
        known_rows = []
        row_idx = 0
        for bbox, track_id in zip(detections.xyxy, detections.tracker_id):
            if track_id != -1:
                if track_id in self.history:
                    known_ids.append(track_id)
                    known_rows.append(row_idx)
                else:
                    new_ids.append(track_id)
                    query_rows.append(row_idx)

                crops.append(frame.crop(bbox))
                row_idx += 1

        if not crops:
            return detections
        
        embeddings_current = self.embedder.embed(crops)
        if not self.history:
            for row_idx, track_id in zip(query_rows, new_ids):
                self.history[track_id] = embeddings_current[row_idx]

            return detections

        history_pool_ids = []
        history_pool_vals = []
        for track_id, embedding in self.history.items():
            if track_id not in known_ids:
                history_pool_ids.append(track_id)
                history_pool_vals.append(embedding)

        if history_pool_vals:
            embeddings_history = torch.stack(history_pool_vals)

            embeddings_current = torch.stack(embeddings_current)
            embs_current = F.normalize(embeddings_current[query_rows], dim=1)
            embs_history = F.normalize(embeddings_history, dim=1)

            sim = embs_current @ embs_history.T
            row_ind, col_ind = linear_sum_assignment(np.asarray(sim.float()), maximize=True)

            matched_queries = set()

            for idx_cur, idx_hist in zip(row_ind, col_ind):
                cos = sim[idx_cur][idx_hist]
                original_row_idx = query_rows[idx_cur]

                if cos > self.threshold:
                    correct_id = history_pool_ids[idx_hist]
                    detections.tracker_id[original_row_idx] = correct_id
                else:
                    correct_id = new_ids[idx_cur]
                
                self.history[correct_id] = embeddings_current[original_row_idx]
                matched_queries.add(idx_cur)

            for idx_cur, original_row_idx in enumerate(query_rows):
                if idx_cur not in matched_queries:
                    correct_id = new_ids[idx_cur]
                    self.history[correct_id] = embeddings_current[original_row_idx]

        for row_idx, track_id in zip(known_rows, known_ids):
            self.history[track_id] = embeddings_current[row_idx]

        return detections


class DeepTrackerACM:  # Appearance - Class - Motion
    def __init__(self, embedder: Embedder, threshold: float):
        self.embedder = embedder
        self.threshold = threshold
        # track_id -> (embedding, class_id)
        self.history: dict[int, tuple[torch.Tensor, str]] = {}

    def update(self, detections: Detections, frame: Image.Image, tracker: BoTSORTTracker):
        new_ids = []
        known_ids = []
        crops = []
        row_classes = []
        query_rows = []
        known_rows = []
        row_idx = 0
        for bbox, track_id, class_id in zip(detections.xyxy, detections.tracker_id, detections.class_id):
            if track_id != -1:
                cls = vocab_mapping(class_id)
                if track_id in self.history:
                    known_ids.append(track_id)
                    known_rows.append(row_idx)
                else:
                    new_ids.append(track_id)
                    query_rows.append(row_idx)

                crops.append(frame.crop(bbox))
                row_classes.append(cls)
                row_idx += 1

        if not crops:
            return detections

        embeddings_current = self.embedder.embed(crops)
        if not self.history:
            for row_idx, track_id in zip(query_rows, new_ids):
                self.history[track_id] = (embeddings_current[row_idx], row_classes[row_idx])
            return detections

        history_pool_ids = []
        history_pool_vals = []
        history_pool_classes = []
        for track_id, (embedding, cls) in self.history.items():
            if track_id not in known_ids:
                history_pool_ids.append(track_id)
                history_pool_vals.append(embedding)
                history_pool_classes.append(cls)

        # tracker_id -> live tracklet, so we can rewrite the tracker's own
        # bookkeeping, not just this frame's output detections
        id_to_tracklet = {t.tracker_id: t for t in tracker.tracks}

        if history_pool_vals:
            matched_queries = set()

            query_rows_by_class: dict[str, list[int]] = defaultdict(list)
            for r in query_rows:
                query_rows_by_class[row_classes[r]].append(r)

            history_idx_by_class: dict[str, list[int]] = defaultdict(list)
            for i, cls in enumerate(history_pool_classes):
                history_idx_by_class[cls].append(i)

            for cls, cls_query_rows in query_rows_by_class.items():
                cls_hist_idx = history_idx_by_class.get(cls, [])
                if not cls_hist_idx:
                    continue

                embs_current = F.normalize(
                    torch.stack([embeddings_current[r] for r in cls_query_rows]), dim=1
                )
                embs_history = F.normalize(
                    torch.stack([history_pool_vals[i] for i in cls_hist_idx]), dim=1
                )

                sim = embs_current @ embs_history.T
                row_ind, col_ind = linear_sum_assignment(np.asarray(sim.float()), maximize=True)

                for idx_cur, idx_hist in zip(row_ind, col_ind):
                    cos = sim[idx_cur][idx_hist]
                    original_row_idx = cls_query_rows[idx_cur]
                    query_global_idx = query_rows.index(original_row_idx)
                    new_id = new_ids[query_global_idx]

                    if cos > self.threshold:
                        correct_id = history_pool_ids[cls_hist_idx[idx_hist]]
                        detections.tracker_id[original_row_idx] = correct_id

                        # Option 1: drop the old (lost) tracklet, keep the new
                        # tracklet's fresher Kalman state, just relabeled.
                        tracklet_new = id_to_tracklet.get(new_id)
                        tracklet_old = id_to_tracklet.get(correct_id)
                        if tracklet_old is not None and tracklet_old is not tracklet_new:
                            tracker.tracks = [t for t in tracker.tracks if id(t) != id(tracklet_old)]
                            id_to_tracklet.pop(correct_id, None)
                        if tracklet_new is not None:
                            tracklet_new.tracker_id = correct_id
                            id_to_tracklet[correct_id] = tracklet_new
                            id_to_tracklet.pop(new_id, None)
                    else:
                        correct_id = new_id

                    self.history[correct_id] = (embeddings_current[original_row_idx], cls)
                    matched_queries.add(original_row_idx)

            for original_row_idx in query_rows:
                if original_row_idx not in matched_queries:
                    query_global_idx = query_rows.index(original_row_idx)
                    correct_id = new_ids[query_global_idx]
                    self.history[correct_id] = (embeddings_current[original_row_idx], row_classes[original_row_idx])

        for row_idx, track_id in zip(known_rows, known_ids):
            self.history[track_id] = (embeddings_current[row_idx], row_classes[row_idx])

        return detections


class DeepTrackerA: # Appearance
    def __init__(self, embedder: Embedder, threshold: float):
        self.embedder = embedder
        self.threshold = threshold
        self.history: dict[int, torch.Tensor] = {}
        # Introduce an explicit counter for ID generation
        self.next_id = 0 

    def update(self, detections: Detections, frame: Image.Image):
        crops = [frame.crop(bbox) for bbox in detections.xyxy]
        
        embeddings_current = self.embedder.embed(crops)
        embeddings_current = torch.stack(embeddings_current)

        # Handle the starting case (empty history)
        if not self.history:
            tracked_ids = []
            for emb in embeddings_current:
                self.history[self.next_id] = emb
                tracked_ids.append(self.next_id)
                self.next_id += 1
            
            # Assign IDs to the detections object before returning
            detections.tracker_id = tracked_ids
            return detections

        # Proceed with normal matching if history exists
        embeddings_history = torch.stack(list(self.history.values()))

        embs_current = F.normalize(embeddings_current, dim=1)
        embs_history = F.normalize(embeddings_history, dim=1)

        sim = embs_current @ embs_history.T
        row_ind, col_ind = linear_sum_assignment(np.asarray(sim.float()), maximize=True)

        history_ids = list(self.history.keys())
        for idx_cur, idx_hist in zip(row_ind, col_ind):
            cos = sim[idx_cur][idx_hist]

            if cos > self.threshold:
                correct_id = history_ids[idx_hist]
                detections.tracker_id[idx_cur] = correct_id
                self.history[correct_id] = embeddings_current[idx_cur]

        return detections
