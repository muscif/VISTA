from PIL import Image
from supervision.detection.core import Detections
from trackers import BoTSORTTracker as Tracker
import torch
import torch.nn.functional as F

from caption import Caption, CaptionerQwen3VL
from utils import _iou, draw_bboxes_single, vocab_mapping, get_smallest_bbox
from base import Detection, VistaPipeline, FrameResult
from tracker import Embedder


class SISTA(VistaPipeline):
    def __init__(self, caption_stride=30):
        from rfdetr_plus import RFDETR2XLarge as RFDETR
        from tracker import DeepTrackerACM as DeepTracker

        self.model = RFDETR()
        self.model.optimize_for_inference(dtype=torch.bfloat16, compile=True)
        self.tracker = Tracker(enable_cmc=True)
        self.captioner = None
        #self.captioner = CaptionerQwen3VL("Qwen/Qwen3-VL-4B-Instruct-FP8")
        self.caption_stride = caption_stride
        self.iou_threshold = 0.5
        self.history = {}
        self.crop = True
        self.draw_bboxes = False
        self.deep_tracker = DeepTracker(Embedder("facebook/dinov2-with-registers-large"), 0.5)
        self.reid = True

    def forward(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        results = self.model.predict(frame)
        results = self.tracker.update(results, frame=results.metadata["source_image"])

        if self.reid:
            results = self.deep_tracker.update(results, frame, self.tracker)

        detections = {res[4]: Detection(res[0].tolist(), vocab_mapping(res[3]), 1, res[4]) for i, res in enumerate(results)}

        if self.draw_bboxes:
            frame = draw_bboxes_single(frame, detections)

        min_x1 = min_y1 = max_x2 = max_y2 = 0

        if self.captioner and frame_idx % self.caption_stride == 0:
            if self.crop:
                min_x1, min_y1, max_x2, max_y2 = get_smallest_bbox(detections)
                frame = frame.crop((min_x1, min_y1, max_x2, max_y2))

            captions: list[Caption] = self.captioner.caption(frame)

            for caption in captions:
                best_iou = 0
                best_id = None

                for detection in detections.values():
                    caption_bbox = (caption.bbox[0] + min_x1, caption.bbox[1] + min_y1, caption.bbox[2] + min_x1, caption.bbox[3] + min_y1)
                    iou = _iou(detection.bbox, caption_bbox)

                    if iou > self.iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_id = detection.track_id

                    if best_id is not None:
                        detections[best_id].caption = caption.caption
                        self.history[best_id] = caption.caption

        for det_id in detections:
            if detections[det_id].caption is None:
                capt = self.history.get(
                    det_id, detections[det_id].category
                )
                detections[det_id].caption = f"{det_id}: {capt}"

        return FrameResult(
            detections=list(detections.values()),
            frame_idx=frame_idx
        )

    def reset(self):
        super().reset()
        self.__init__(self.caption_stride)
    
