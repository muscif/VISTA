from abc import ABC, abstractmethod

import bbox_visualizer as bbv
from PIL.Image import Image

from base import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, img: Image, track_db: dict[int, str]) -> dict[int, Detection]:
        pass


class DetectorYOLO(Detector):
    def __init__(self, yolo_name, imgsz):
        from ultralytics import YOLO
        self.detector: YOLO = YOLO(yolo_name)
        self.imgsz = imgsz

    def detect(self, img: Image, track_db: dict[int, str]) -> dict[int, Detection]:
        results = self.detector.track(
            img, persist=True, verbose=False, end2end=False, iou=0.3, conf=0.1, imgsz=self.imgsz
        )[0]

        detections = {}
        for box, track_id, cls in zip(results.boxes.xyxy, results.boxes.id, results.boxes.cls):
            tid = int(track_id.item())
            cat = results.names.get(int(cls.item()), "unknown")

            bbox = tuple(box.cpu().numpy().tolist())
            category = track_db.get(tid, {}).get("category", cat)
            conf = float(results.boxes.conf[list(results.boxes.id).index(tid)].item())

            d = Detection(bbox, category, conf, tid, None)
            detections[tid] = d

        return detections
