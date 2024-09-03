class PaddleDetectionConfig:
    # 模型目录
    MODEL_DIR = "models/ppyoloe_plus_crn_s_80e_coco"

    # 预测阈值
    THRESHOLD = 0.5


class BiliDownloadConfig:
    # 视频下载目录
    PATH = "download/video"

    # 分段下载并发数
    PART_CONCURRENCY = 1

    # 视频下载并发数
    CONCURRENCY = 1

    # 视频下载时间范围（秒）
    TIME_RANGE = (0, 10)


class VideoProcessConfig:
    # 视频帧截取间隔（秒）
    CAPTURE_EVERY = 1
