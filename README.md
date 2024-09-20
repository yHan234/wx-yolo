# 文心悠洛视频分析智能体

使用 [flask](https://github.com/pallets/flask) 构建服务器，为[文心智能体](https://agents.baidu.com)提供接口，结合 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 开发套件进行目标检测，让文心智能体拥有分析视频的能力。

工作流程：用户向文心智能体发送 [BiliBili](https://www.bilibili.com/) 链接并要求分析视频，文心智能体调用设置好的工作流，该工作流会使用 [bilix](https://github.com/HFrost0/bilix) 下载视频，使用 [opencv-python](https://github.com/opencv/opencv-python) 从视频中截取帧，使用 [PP-YOLOE 模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.7/configs/ppyoloe/README_cn.md)进行目标检测，将检测结果返回给文心智能体，文心智能体总结后回复用户。
