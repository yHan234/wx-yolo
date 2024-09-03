import os
from bilix.sites.bilibili import DownloaderBilibili
from config import BiliDownloadConfig
from utils.video_file import find_video

downloader = DownloaderBilibili(
    part_concurrency=BiliDownloadConfig.PART_CONCURRENCY,
    video_concurrency=BiliDownloadConfig.CONCURRENCY,
)


async def bili_download(url: str) -> str:
    """下载 bilibili 视频，返回下载后的视频文件的路径
    可在config.py:BiliDownloadConfig中修改下载配置"""
    print("开始下载视频: " + url)

    path = os.path.join(BiliDownloadConfig.PATH, "bili", url[31 : 31 + 12])

    if not os.path.exists(path):
        os.makedirs(path)
    elif (video_path := find_video(path)) is not None:
        print("视频已缓存: " + video_path)
        return video_path

    await downloader.get_video(
        url,
        path,
        time_range=BiliDownloadConfig.TIME_RANGE,
    )

    print("视频下载完成: ", video_path := find_video(path))
    return video_path
