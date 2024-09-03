import cv2
import os
from config import VideoProcessConfig


def capture_frames(video_path):
    """抓取视频帧"""
    print("开始抓取视频帧:" + video_path)

    video_dir = os.path.dirname(video_path)
    output_dir = os.path.join(video_dir, "frames")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Open video file failed.")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(int(fps * VideoProcessConfig.CAPTURE_EVERY), 1)
    frame_count = 0
    out_paths = []
    out_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            time = round(frame_count / fps, 2)
            output_file = os.path.join(output_dir, f"frame_{time}.jpg")
            cv2.imwrite(output_file, frame)

            out_paths = out_paths + [output_file]
            out_times = out_times + [time]

        frame_count += 1

    cap.release()
    return out_paths, out_times
