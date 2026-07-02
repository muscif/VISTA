import os
from pathlib import Path
import time
import tomllib

import bbox_visualizer as bbv
import cv2
from PIL import Image

from detect import Detector, DetectorYOLO
from caption import Captioner, CaptionerQwen3VL
from sista_whole import SISTA
from base import FrameResult, Detection

PATH_BASE = Path(".")
PATH_DATA = PATH_BASE / "data"

PATH_VISTA = PATH_DATA / "VISTADataset"

os.environ["HF_HUB_DISABLE_XET"] = "1"


def draw_bboxes(path: Path, frames: list[FrameResult]):
    vid = cv2.VideoCapture(path)

    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or "avc1", "XVID" depending on codec support
    out = cv2.VideoWriter("out.mkv", fourcc, fps, (width, height))

    success = True
    i_frame = 0

    while success:
        success, image = vid.read()

        if success:
            res = frames[i_frame].detections

            bboxes = []
            labels = []
            for d in res:
                bboxes.append([int(n) for n in d.bbox])
                labels.append(d.caption)

            img = bbv.draw_multiple_boxes(image, bboxes)
            img = bbv.add_multiple_labels(img, labels, bboxes, size=0.4, thickness=1)

            out.write(img)

        i_frame += 1

        if i_frame >= len(frames):
            break

    vid.release()
    out.release()


if __name__=="__main__":
    detector: Detector = DetectorYOLO("yolo26x_visdrone.pt", 800)
    captioner: Captioner = CaptionerQwen3VL("Qwen/Qwen3-VL-4B-Instruct-FP8")

    model = SISTA(detector, captioner, 60, 0.3)

    path = PATH_VISTA / "train" / "20251120" / "DJI_20251120172410_0001_S.mp4"

    start_frame = 4320
    duration = 360
    
    t0 = time.perf_counter()
    results = [res for res in model.process_video(path, start_frame=start_frame, end_frame=start_frame + duration)]
    elapsed = time.perf_counter() - t0

    frame_count = len(results)

    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"Processed {frame_count} frames in {elapsed:.2f}s -> {fps:.2f} FPS")

    draw_bboxes(path, results)