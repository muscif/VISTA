from pathlib import Path

import cv2
import bbox_visualizer as bbv
import numpy as np
from PIL import Image

from base import FrameResult, Detection

def _iou(a, b) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)

    if inter <= 0:
        return 0.0
    aA = (a[2] - a[0]) * (a[3] - a[1])
    aB = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (aA + aB - inter)

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


def draw_bboxes_single(frame: Image.Image, detections: dict[int, Detection]) -> Image.Image:
    bboxes = []
    for det in detections.values():
        bboxes.append([int(n) for n in det.bbox])

    
    img = bbv.draw_multiple_boxes(np.asarray(frame), bboxes)
    return Image.fromarray(img)


def draw_bboxes(path: Path, frames: list[FrameResult]):
    if not frames:
        return
        
    vid = cv2.VideoCapture(str(path))
    
    # Seek to the exact frame where the results start
    start_frame = frames[0].frame_idx
    if start_frame > 0:
        vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("out.mkv", fourcc, fps, (width, height))

    success = True
    i_frame = 0

    while success:
        success, image = vid.read()

        if success:
            res = frames[i_frame].detections
            img = image

            if len(res) > 0:
                bboxes = []
                labels = []

                for d in res:
                    bboxes.append([int(n) for n in d.bbox])
                    labels.append(d.caption)

                img = bbv.draw_multiple_boxes(img, bboxes)
                img = bbv.add_multiple_labels(img, labels, bboxes, size=0.4, thickness=1)

            out.write(img)

        i_frame += 1

        if i_frame >= len(frames):
            break

    vid.release()
    out.release()


def prediction_tracks(video_frames: dict[str, list[FrameResult]]):
    for video_id, frame_results in video_frames:
        pass


def predictions_mot(video_frames: dict[str, list[FrameResult]]):
    for video_id, frame_results in video_frames:
        pass

def vocab_mapping(predict) -> str:
    match predict:
        case 1:
            return "person"
        case 2 | 3 | 4 | 5 | 6 | 7 | 8:
            return "vehicle"
        case _:
            return "other"
        

def get_smallest_bbox(detections: dict[int, Detection]) -> tuple[float, float, float, float]:
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


def expand_bbox(bbox, amount: float) -> tuple[float, float, float, float]:
    assert -1 <= amount <= 1, "Amount must be in [-1, 1]"
    return bbox[0] * 1-amount, bbox[1] * 1-amount, bbox[2] * 1+amount, bbox[3] * 1+amount
