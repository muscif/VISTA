import os
from pathlib import Path
import time

from tqdm import tqdm

from cat_vista import SISTA
from utils import draw_bboxes, PATH_VISTA, prediction_tracks, predictions_mot, compute_fps

os.environ["HF_HUB_DISABLE_XET"] = "1"


if __name__=="__main__":
    tracker_name = "acm"
    model = SISTA(tracker_name=tracker_name, caption_stride=60, caption=False)

    things = [
        #(PATH_VISTA / "train" / "20251120" / "DJI_20251120172410_0001_S.mp4", 12500, 12780),
        (PATH_VISTA / "train" / "20251205" / "DJI_20251205134959_0001_S.mp4", 2320, 3070),
        #(PATH_VISTA / "train" / "20251204" / "DJI_20251204135749_0001_S.mp4", 676, 740),
        #(PATH_VISTA / "train" / "20260318" / "DJI_20260318101426_0002_V.MP4", 5800, 7300),
        #(PATH_VISTA / "train" / "20260318" / "DJI_20260318102010_0003_V.MP4", 4778, 6128)
    ]

    videos = {}
    fps_stats = {}
    for path, start_frame, end_frame in things:
        results = []
        timings = []

        total = end_frame - start_frame
        
        start = time.perf_counter()
        for res in tqdm(model.process_video(path, start_frame=start_frame, end_frame=end_frame), leave=False, total=end_frame-start_frame):
            end = time.perf_counter()
            timings.append(end - start)
            results.append(res)
            start = time.perf_counter()

        frame_count = len(results)

        video_id = path.parts[-1].removesuffix(path.suffix)
        videos[video_id] = results
        fps_stats[video_id] = timings
        draw_bboxes(path, results, tracker_name)

    compute_fps(fps_stats)
    prediction_tracks(videos, "prediction_tracks_train.csv")
    predictions_mot(videos, "predictions_mot_train.csv")

