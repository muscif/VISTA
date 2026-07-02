from base import VistaPipeline, FrameResult, Detection
from PIL import Image

from detect import Detector
from caption import Captioner, Caption


def _iou(a, b) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)

    if inter <= 0:
        return 0.0
    aA = (a[2] - a[0]) * (a[3] - a[1])
    aB = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (aA + aB - inter)


class SISTA(VistaPipeline):
    def __init__(self, detector: Detector, captioner: Captioner, caption_stride: int, iou_threshold: float):
        self.detector = detector
        self.captioner = captioner
        self.caption_stride = caption_stride
        self.track_db = {}
        self.history = {}
        self.iou_threshold = iou_threshold

    def forward(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        detections: dict[int, Detection] = self.detector.detect(frame, self.track_db)

        if self.captioner and frame_idx % self.caption_stride == 0:
            captions: list[Caption] = self.captioner.caption(frame)

            for caption in captions:
                best_iou = 0
                best_id = None

                for detection in detections.values():
                    iou = _iou(detection.bbox, caption.bbox)

                    if iou > self.iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_id = detection.track_id

                if best_id is not None:
                    detections[best_id].caption = caption.caption
                    self.history[best_id] = caption.caption  # remember it

        for det_id in detections:
            if detections[det_id].caption is None:
                detections[det_id].caption = self.history.get(det_id, detections[det_id].category)

        return FrameResult(
            detections=list(detections.values()),
            frame_idx=frame_idx,
        )

    def reset(self) -> None:
        self.track_db.clear()
        self.history.clear()
