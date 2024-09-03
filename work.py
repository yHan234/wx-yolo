from video_download import *
from video_process import *


async def download_and_detect(url: str):
    if not url.startswith("https://www.bilibili.com/video/BV"):
        raise Exception(
            r'提供的视频网址错误，应当以"https://www.bilibili.com/video/BV" 开头'
        )

    url = url[:43]
    video_path = await bili_download(url)

    image_list, time_list = capture_frames(video_path)

    frames = object_detect(image_list)

    for i, frame in enumerate(frames):
        frame["time"] = time_list[i]

    return frames
