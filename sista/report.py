import time

from tqdm import tqdm

from cat_vista import SISTA
from utils import draw_bboxes, PATH_VISTA, prediction_tracks, predictions_mot, compute_fps

if __name__=="__main__":
    tracker_name = "acm"
    model = SISTA(tracker_name=tracker_name, caption_stride=300, caption=True)

    things = [
        #(PATH_VISTA / "test" / "20251210" / "DJI_20251210134636_0001_S.mp4", 0, 6695),
        #(PATH_VISTA / "test" / "20251210" / "DJI_20251210140457_0001_S.mp4", 0, 9320),
        (PATH_VISTA / "test" / "20251217" / "DJI_20251217111534_0001_S.mp4", 0, 25874),
        #(PATH_VISTA / "test" / "20251217" / "DJI_20251217114349_0001_S.MP4", 0, 9237),
    ]


    for path, start_frame, end_frame in things:
        videos = {}
        fps_stats = {}
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

        prediction_tracks(videos, fout=f"predictions_tracks_{video_id}.csv")
        predictions_mot(videos, fout=f"predictions_mot_{video_id}.csv")
        compute_fps(fps_stats, fout=f"fps_stats_{video_id}.csv")
