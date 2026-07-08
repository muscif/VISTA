import unsloth

from PIL import Image
from supervision.detection.core import Detections
from trackers import BoTSORTTracker as Tracker
from rfdetr.assets.coco_classes import COCO_CLASSES
import torch
import torch.nn.functional as F

from caption import Caption, CaptionerQwen3VL
from utils import _iou, draw_bboxes_single, get_smallest_bbox
from base import Detection, VistaPipeline, FrameResult
from tracker import Embedder


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
            case "am":
                from tracker import DeepTrackerAM

                self.deep_tracker = DeepTrackerAM(embedder, reid_similarity_threshold)
            case "acm":
                from tracker import DeepTrackerACM

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

        if self.captioner and frame_idx % self.caption_stride == 0 and detections:
            min_x1 = min_y1 = max_x2 = max_y2 = 0
            
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
