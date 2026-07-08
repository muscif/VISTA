from collections import defaultdict
from dataclasses import dataclass

import bbox_visualizer as bbv
from PIL import Image
import json
import numpy as np
from rfdetr.assets.coco_classes import COCO_CLASSES
from scipy.optimize import linear_sum_assignment
from supervision import Detections
import torch
import torch.nn.functional as F
from trackers import BoTSORTTracker as Tracker

from base import VistaPipeline, Detection, FrameResult

system_prompt = """
    You are an operator supervising a drone operation over an accident scene. Your task is to detect and label all relevant objects in the images. Focus on the following:

    1. Vehicles:
      - Identify and classify all vehicles, including cars, trucks, motorcycles, bicycles only if they are involved in the accident, ignore the rest.
      - Distinguish between:
        * Vehicles involved in the accident
        * Emergency or helping vehicles

    2. People:
      - Detect all people present in the scene.
      - Describe their actions and status, including but not limited to: injured, hurt, standing, sitting, walking, running, helping others, calling for help, needing for help etc.
      - Include this information in the label.

    Output format:
    - Return a valid JSON array with bounding boxes for all detected elements in the form:
      `[{"bbox_2d": [xmin, ymin, xmax, ymax], "label": "detailed description"}, ...]`
    - Example valid response:
      `[{"bbox_2d": [10, 30, 20, 60], "label": "car involved in accident"}, {"bbox_2d": [40, 15, 52, 27], "label": "person injured, sitting"}]`
    - Ensure each object is labeled with a precise description reflecting its type and status.
"""

user_prompt = """
    Detect and label all relevant vehicles and persons in this frame.
"""


def vocab_mapping(predict) -> str:
    match predict:
        case 1:
            return "person"
        case 2 | 3 | 4 | 5 | 6 | 7 | 8:
            return "vehicle"
        case _:
            return "other"


def postprocess_boxes(data, img):
    width, height = img.size
    for item in data:
        x1, y1, x2, y2 = item["bbox_2d"]
        item["bbox_2d"] = [
            x1 / 1000 * width,
            y1 / 1000 * height,
            x2 / 1000 * width,
            y2 / 1000 * height,
        ]
    return data


def draw_bboxes_single(
    frame: Image.Image, detections: dict[int, Detection]
) -> Image.Image:
    bboxes = []
    for det in detections.values():
        bboxes.append([int(n) for n in det.bbox])

    img = bbv.draw_multiple_boxes(np.asarray(frame), bboxes)
    return Image.fromarray(img)


def get_smallest_bbox(
    detections: dict[int, Detection],
) -> tuple[float, float, float, float]:
    min_x1 = float("inf")
    min_y1 = float("inf")
    max_x2 = 0
    max_y2 = 0

    for d in detections.values():
        x1, y1, x2, y2 = d.bbox

        if x1 < min_x1:
            min_x1 = x1
        if y1 < min_y1:
            min_y1 = y1
        if x2 > max_x2:
            max_x2 = x2
        if y2 > max_y2:
            max_y2 = y2

    return int(min_x1), int(min_y1), int(max_x2), int(max_y2)


def _iou(a, b) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)

    if inter <= 0:
        return 0.0
    aA = (a[2] - a[0]) * (a[3] - a[1])
    aB = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (aA + aB - inter)


@dataclass
class Caption:
    bbox: tuple[float, float, float, float]
    caption: str


class CaptionerQwen3VL:
    def __init__(self, model_name):
        from unsloth import FastVisionModel

        self.model, processor = FastVisionModel.from_pretrained(
            model_name=model_name, load_in_4bit=True
        )

        FastVisionModel.for_inference(self.model)

        self.processor = processor
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
    
    def caption(self, img: Image.Image):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": self.user_prompt},
            ]},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda:0")

        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=8192)

        gen_ids = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
        raw_output = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        cleaned_output = raw_output.strip()
        
        # Strip markdown code blocks if present
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output.split("```json")[1].split("```")[0].strip()
        elif cleaned_output.startswith("```"):
            cleaned_output = cleaned_output.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(cleaned_output)
            data = postprocess_boxes(data, img)
            
            captions = []
            for item in data:
                # Convert the list of 4 floats into a tuple as required by Caption dataclass
                bbox_tuple = tuple(item["bbox_2d"])
                
                if len(bbox_tuple) == 4:
                    captions.append(
                        Caption(
                            bbox=bbox_tuple,  # type: ignore
                            caption=item["label"]
                        )
                    )
            return captions
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            # Fallback/Error handling if the model generates malformed JSON
            print(f"Failed to parse model response to JSON: {e}")
            print(f"Raw response was: {raw_output}")
            return []


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


class DeepTrackerACM:  # Appearance - Class - Motion
    def __init__(self, embedder: Embedder, threshold: float):
        self.embedder = embedder
        self.threshold = threshold
        self.history: dict[int, tuple[torch.Tensor, str]] = {}

    def update(self, detections: Detections, frame: Image.Image, tracker: Tracker):
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
    
    def reset(self):
        self.history = {}


class SISTA(VistaPipeline):
    def __init__(
        self,
        caption_stride=30,
        tracker_name=None,
        embedder_name="facebook/dinov2-with-registers-large",
        caption_iou_threshold=0.5,
        reid_similarity_threshold=0.5,
        caption=False,
    ):
        from rfdetr_plus import RFDETR2XLarge as RFDETR
        # from rfdetr import RFDETRLarge as RFDETR

        embedder = Embedder(embedder_name)

        self.embedder_name = embedder_name
        self.tracker_name = tracker_name
        self.reid_similarity_threshold = reid_similarity_threshold

        match self.tracker_name:
            case "acm":
                self.deep_tracker = DeepTrackerACM(embedder, reid_similarity_threshold)
            case _:
                self.deep_tracker = None

        self.model = RFDETR()
        self.model.optimize_for_inference(dtype=torch.bfloat16, compile=True)
        self.tracker = Tracker(enable_cmc=True)
        self.caption = caption
        self.captioner = None
        if self.caption:
            self.captioner = CaptionerQwen3VL("Qwen/Qwen3-VL-4B-Instruct-FP8")
        self.caption_stride = caption_stride
        self.caption_iou_threshold = caption_iou_threshold
        self.history = {}
        self.crop = True
        self.draw_bboxes = False

    def forward(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        results = self.model.predict(frame)
        results = self.tracker.update(results)#, frame=results.metadata["source_image"])

        if self.deep_tracker:
            results = self.deep_tracker.update(results, frame, self.tracker)

        detections = {
            int(res[4]): Detection(
                bbox=res[0].tolist(), category=COCO_CLASSES[res[3]], confidence=res[2], track_id=int(res[4])
            )
            for i, res in enumerate(results)
            if int(res[4]) != -1
        }

        if self.draw_bboxes:
            frame = draw_bboxes_single(frame, detections)

        min_x1 = min_y1 = max_x2 = max_y2 = 0

        if self.captioner and frame_idx % self.caption_stride == 0 and len(results.xyxy) > 0:
            if self.crop:
                min_x1, min_y1, max_x2, max_y2 = get_smallest_bbox(detections)
                frame = frame.crop((min_x1, min_y1, max_x2, max_y2))

            captions: list[Caption] = self.captioner.caption(frame)

            for caption in captions:
                best_iou = 0
                best_id = None

                for detection in detections.values():
                    caption_bbox = (
                        caption.bbox[0] + min_x1,
                        caption.bbox[1] + min_y1,
                        caption.bbox[2] + min_x1,
                        caption.bbox[3] + min_y1,
                    )
                    iou = _iou(detection.bbox, caption_bbox)

                    if iou > self.caption_iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_id = detection.track_id

                    if best_id is not None:
                        detections[best_id].caption = caption.caption
                        self.history[best_id] = caption.caption

        for det_id in detections:
            if detections[det_id].caption is None:
                capt = self.history.get(det_id, detections[det_id].category)
                detections[det_id].caption = f"{det_id}: {capt}"

        return FrameResult(detections=list(detections.values()), frame_idx=frame_idx)

    def reset(self):
        super().reset()
        self.history = {}
        self.tracker = Tracker(enable_cmc=True)
        if self.deep_tracker:
            self.deep_tracker.reset()
