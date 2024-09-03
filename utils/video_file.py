import os
import mimetypes


def is_video_file(path: str) -> bool:
    """通过 mimetypes.guess_type 来判断文件是否是视频文件"""
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is not None:
        return mime_type.startswith("video/")
    return False


def find_video(dir: str) -> str:
    """返回目录下第一个视频文件的路径"""
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        if os.path.isfile(filepath) and is_video_file(filepath):
            return filepath
