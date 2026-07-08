from collections import defaultdict
import csv
from pathlib import Path

import cv2
import bbox_visualizer as bbv
import numpy as np
from PIL import Image

from base import FrameResult, Detection


PATH_BASE = Path(".")
PATH_DATA = PATH_BASE / "data"
PATH_VISTA = PATH_DATA / "VISTADataset"


def _iou(a, b) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)

    if inter <= 0:
        return 0.0
    aA = (a[2] - a[0]) * (a[3] - a[1])
    aB = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (aA + aB - inter)


def compute_fps(fps_stats: dict[str, list]):
    rows = []
    for video_id, timings in fps_stats.items():
        timings = timings[1:]  # exclude first frame (setup/warmup overhead)
        fps_values = [1.0 / t for t in timings if t > 0]
        min_fps = min(fps_values)
        avg_fps = sum(fps_values) / len(fps_values)

        rows.append((video_id, min_fps, avg_fps))

    with open("fps_stats.csv", "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["video_id", "fps_min", "fps_avg"])
        writer.writerows(rows)


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


def draw_bboxes(path: Path, frames: list[FrameResult], deep_tracker):
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
    video_id = path.parts[-1].removesuffix(path.suffix)
    out = cv2.VideoWriter(
        f"out/{video_id}_{deep_tracker}.mkv", fourcc, fps, (width, height)
    )

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
                img = bbv.add_multiple_labels(
                    img, labels, bboxes, size=0.4, thickness=1
                )

            out.write(img)

        i_frame += 1

        if i_frame >= len(frames):
            break

    vid.release()
    out.release()


# video_id, track_id, frame_start, frame_end, caption
def prediction_tracks(video_frames: dict[str, list[FrameResult]], fout="out.csv"):
    video_tracks = defaultdict(list)
    for video_id, frame_results in video_frames.items():
        for frame_result in frame_results:
            for detection in frame_result.detections:
                video_tracks[(video_id, detection.track_id)].append(
                    (frame_result.frame_idx, detection.caption.split(":")[-1].strip())
                )

    rows = []
    for (video_id, track_id), els in video_tracks.items():
        els_sorted = sorted(els, key=lambda x: x[0])

        interval_start, interval_caption = els_sorted[0]
        interval_end = interval_start
        prev_frame = interval_start

        intervals = []
        for frame_idx, caption in els_sorted[1:]:
            if frame_idx == prev_frame + 1:
                interval_end = frame_idx
                interval_caption = caption
            else:
                intervals.append((interval_start, interval_end, interval_caption))
                interval_start, interval_caption = frame_idx, caption
                interval_end = frame_idx
            prev_frame = frame_idx

        intervals.append((interval_start, interval_end, interval_caption))  # flush last

        for interval_start, interval_end, caption in intervals:
            rows.append((video_id, track_id, interval_start, interval_end, caption))

    with open(fout, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["video_id", "track_id", "frame_start", "frame_end", "caption"])
        writer.writerows(rows)


# video_id, frame_id, track_id, x1, y1, x2, y2, conf, category
def predictions_mot(video_frames: dict[str, list[FrameResult]], fout="predictions_mot.csv"):
    rows = []
    for video_id, frame_results in video_frames.items():
        for frame in frame_results:
            frame_id = frame.frame_idx

            for detection in frame.detections:
                track_id = detection.track_id
                x1, y1, x2, y2 = detection.bbox
                conf = detection.confidence
                category = detection.category

                tup = video_id, frame_id, track_id, x1, y1, x2, y2, conf, category
                rows.append(tup)

    with open(fout, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["video_id", "frame_id", "track_id", "x1", "y1", "x2", "y2", "conf", "category"])
        writer.writerows(rows)


def vocab_mapping(predict) -> str:
    match predict:
        case 1:
            return "person"
        case 2 | 3 | 4 | 5 | 6 | 7 | 8:
            return "vehicle"
        case _:
            return "other"


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


def expand_bbox(bbox, amount: float) -> tuple[float, float, float, float]:
    assert -1 <= amount <= 1, "Amount must be in [-1, 1]"
    return (
        bbox[0] * 1 - amount,
        bbox[1] * 1 - amount,
        bbox[2] * 1 + amount,
        bbox[3] * 1 + amount,
    )
