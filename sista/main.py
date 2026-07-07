import os
from pathlib import Path
import time

from cat_vista import SISTA
from utils import draw_bboxes
from tqdm import tqdm

PATH_BASE = Path(".")
PATH_DATA = PATH_BASE / "data"

PATH_VISTA = PATH_DATA / "VISTADataset"

os.environ["HF_HUB_DISABLE_XET"] = "1"


if __name__=="__main__":
    model = SISTA()

    #path, start_frame, end_frame = PATH_VISTA / "train" / "20251120" / "DJI_20251120172410_0001_S.mp4", 12500, 12780
    path, start_frame, end_frame = PATH_VISTA / "train" / "20251205" / "DJI_20251205134959_0001_S.mp4", 2320, 3070
    #path, start_frame, end_frame = PATH_VISTA / "train" / "20251204" / "DJI_20251204135749_0001_S.mp4", 676, 740
    #path, start_frame, end_frame = PATH_VISTA / "train" / "20260318" / "DJI_20260318101426_0002_V.MP4", 5800, 7300
    #path, start_frame, end_frame = PATH_VISTA / "train" / "20260318" / "DJI_20260318102010_0003_V.MP4", 4778, 6128

    results = []
    t0 = time.perf_counter()
    for res in tqdm(model.process_video(path, start_frame=start_frame, end_frame=end_frame), leave=False, total=end_frame-start_frame):
        results.append(res)
    elapsed = time.perf_counter() - t0

    frame_count = len(results)

    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"Processed {frame_count} frames in {elapsed:.2f}s -> {fps:.2f} FPS")

    draw_bboxes(path, results)
