import os
from pathlib import Path
import tomllib

import bbox_visualizer as bbv
import cv2
from PIL import Image

from detect import Detector, DetectorYOLO

PATH_BASE = Path(".")
PATH_DATA = PATH_BASE / "data"

PATH_VISTA = PATH_DATA / "VISTADataset"

os.environ["HF_HUB_DISABLE_XET"] = "1"

if __name__=="__main__":
    detector: Detector = DetectorYOLO("yolo26x_visdrone.pt")

    path = PATH_VISTA / "train" / "20251120" / "DJI_20251120172410_0001_S.mp4"
    vid = cv2.VideoCapture(path)

    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or "avc1", "XVID" depending on codec support
    out = cv2.VideoWriter("out.mkv", fourcc, fps, (width, height))

    success = True
    i_frame = 0

    check_frame = 1000
    while success:
        success, image = vid.read()

        if success and i_frame == check_frame:
            res = detector.detect(image, {})

            bboxes = []
            labels = []
            for d in res.values():
                bboxes.append([int(n) for n in d.bbox])
                labels.append(d.category)

            img = bbv.draw_multiple_boxes(image, bboxes)
            img = bbv.add_multiple_labels(img, labels, bboxes)

            out.write(img)
            break

        i_frame += 1

    vid.release()
    out.release()