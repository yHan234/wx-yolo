# 开发教程

## 项目介绍

使用 [flask](https://github.com/pallets/flask) 构建服务器，为[文心智能体](https://agents.baidu.com)提供接口，结合 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 开发套件进行目标检测，让文心智能体拥有分析视频的能力。

插件工作流程：用户向文心智能体发送 [BiliBili](https://www.bilibili.com/) 链接并要求分析视频，文心智能体调用插件接口，该插件会使用 [bilix](https://github.com/HFrost0/bilix) 下载视频，使用 [opencv-python](https://github.com/opencv/opencv-python) 从视频中截取帧，使用 [PP-YOLOE 模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.7/configs/ppyoloe/README_cn.md)进行目标检测，将检测结果返回给文心智能体，文心智能体总结后回复用户。

## 平台简介

### PaddleDetection & PP-YOLOE

PaddleDetection 是一个基于 PaddlePaddle 的目标检测端到端开发套件，在提供丰富的模型组件和测试基准的同时，注重端到端的产业落地应用，通过打造产业级特色模型|工具、建设产业应用范例等手段，帮助开发者实现数据准备、模型选型、模型训练、模型部署的全流程打通，快速进行落地应用。

PP-YOLOE 是基于 PP-YOLOv2 的卓越的单阶段 Anchor-free 模型，超越了多种流行的 YOLO 模型。PP-YOLOE 避免了使用诸如 Deformable Convolution 或者 Matrix NMS 之类的特殊算子，以使其能轻松地部署在多种多样的硬件上。其使用大规模数据集 obj365 预训练模型进行预训练，可以在不同场景数据集上快速调优收敛。

### 文心智能体平台

文心智能体平台 AgentBuilder 是百度推出的基于文心大模型的智能体（Agent）平台，支持广大开发者根据自身行业领域、应用场景，选取不同类型的开发方式，打造大模型时代的产品能力。开发者可以通过 prompt 编排的方式低成本开发智能体（Agent），同时，文心智能体平台还将为智能体（Agent）开发者提供相应的流量分发路径，完成商业闭环。

## 调试 PaddleDetection PP-YOLOE 模型

用于训练/评估/导出 PP-YOLOE 模型，如果你对这些不感兴趣，可以跳过这一节使用我导出的模型，也不需要安装这些环境。

由于我在安装时遇到了一些小问题，以下步骤结合了我解决问题的过程。

1. 安装 PaddlePaddle v2.3.2 (CPU): `python -m pip install paddlepaddle==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple`（如果你有使用 GPU 的需求，可以阅读 PaddlePaddle 相关文档）
1. 下载 [PaddleDetection v2.7.0](https://github.com/PaddlePaddle/PaddleDetection/releases/tag/v2.7.0) 并解压
1. 根据 [PaddleDetection issue #9016](https://github.com/PaddlePaddle/PaddleDetection/issues/9016)，注释掉 requirements.txt 中的 lap: `# lap`
1. 安装 PaddleDetection 的依赖：`pip install -r requirements.txt`
1. 安装 lap: `pip install lap`
1. 安装 paddledet: `python setup.py install`
1. 测试安装是否成功：`python ppdet/modeling/tests/test_architectures.py`，测试通过后会提示如下信息：`Ran 7 tests in ...s OK`

接下来可以按照 [PaddleDetection README](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.7/README_cn.md) 以及 [PP-YOLOE README](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.7/configs/ppyoloe/README_cn.md) 调试及导出你的模型。

## 视频处理

这一节涉及的文件树：

```
wx-yolo
│  config.py
│  work.py
│
├─models
│  └─...
│
├─utils
│     video_file.py
│
├─video_download
│     bili_download.py
│     __init__.py
│
└─video_process
   │  object_detect.py
   │  video_capture.py
   │  __init__.py
   │
   └─paddle_detection
         infer.py
         ...
```

其中 `config.py` 和 `utils/` 比较简单，且不是项目主要功能，以下不做讲解，可以阅读源代码进行了解。

以下的 pip 安装部分，可以直接使用本项目目录的 requirements.txt 执行 `pip install -r requirements.txt` 来安装。

### 部署 PaddleDetection 模型

从这里开始，我们要开始插件的开发。新建文件夹 `wx-yolo` 作为我们的项目根目录，并新建一个 Python 环境，本项目使用 Python 3.10。

首先安装 PaddlePaddle v2.6.1 (CPU): `pip install paddlepaddle==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple`（如果你有使用 GPU 的需求，可以阅读 PaddlePaddle 相关文档），并安装部署模型需要的依赖 `pip install PyYAML opencv-python==4.5.5.64 numpy==1.23.5 scipy imgaug==0.4.0`

下载 [PaddleDetection v2.7.0 源代码](https://github.com/PaddlePaddle/PaddleDetection/releases/tag/v2.7.0)，将其中的 `deploy/python` 目录，复制并重命名到我们项目的 `video_process/paddle_detection` 目录。

其中的 `infer.py` 是目标检测脚本目录，我们要用到其中的 `Detector` 类和它的 `predict_image` 方法。为什么不是 `predict_video`？因为整个视频的帧太多了，我们只要从视频里截几帧检测就行。

现在我们在 `video_process` 目录下新建文件 `object_detect.py`，写入我们的目标检测代码。你需要把准备好的模型路径传给 Detector，我使用的模型放在了项目中的 `models` 文件夹。

```py
import paddle
from .paddle_detection.infer import Detector
from config import PaddleDetectionConfig

paddle.enable_static()  # infer.py:1216

detector = Detector(model_dir=PaddleDetectionConfig.MODEL_DIR)
labels = detector.pred_config.labels


def object_detect(image_list: list[str]):
    print("开始目标检测")
    results = detector.predict_image(image_list, visual=False)

    # 后处理，参考 Detector:save_coco_results 及 infer.py:visualize
    bbox_results = []
    idx = 0
    for box_num in results["boxes_num"]:
        if "boxes" in results:
            boxes = results["boxes"][idx : idx + box_num]
            expect_boxes = (boxes[:, 1] > PaddleDetectionConfig.THRESHOLD) & (
                boxes[:, 0] > -1
            )
            boxes = boxes[expect_boxes, :]
            bbox_results.append(
                {
                    "objs": [
                        {
                            "obj": labels[int(box[0])],
                            "xywh": [
                                int(box[2]),
                                int(box[3]),
                                int(box[4] - box[2]),
                                int(box[5] - box[3]),
                            ],  # xyxy -> xywh
                        }
                        for box in boxes
                    ]
                }
            )
        idx += box_num

    return bbox_results
```

如果对目标检测有其它需求（使用 GPU，修改参数等），可以自行阅读 PaddleDetection 代码。

### 下载视频

本项目使用 [bilix](https://github.com/HFrost0/bilix) 实现了 BiliBili 视频下载，若有其它需求，可以自行实现。

首先你需要在系统上安装好 `ffmpeg`，然后安装 bilix `pip install bilix`，这时候 bilix 和 paddlepaddle 依赖的 protobuf 版本冲突，我们需要设置系统环境变量 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`

接下来在 `video_download/bili_download.py` 写入我们的下载视频代码：

```py
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
```

### 抓取视频帧

在 `video_process` 新建 `video_capture.py` 写入抓取帧的代码：

使用 opencv 遍历每一帧，每隔固定的间隔保存一帧。

```py
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
```

### 整合以上过程

在项目根目录新建 `work.py`，将以上过程（下载视频->抓取帧->目标检测）封装到一个函数 `download_and_detect`：

```py
from video_download import *
from video_process import *


async def download_and_detect(url: str):
    if not url.startswith("https://www.bilibili.com/video/BV"):
        raise Exception(
            r'提供的视频网址错误，应当以"https://www.bilibili.com/video/BV" 开头'
        )

    video_path = await bili_download(url)

    image_list, time_list = capture_frames(video_path)

    frames = object_detect(image_list)

    for i, frame in enumerate(frames):
        frame["time"] = time_list[i]

    return frames

```

## 文心智能体插件

这一节涉及的文件树：

```
wx-yolo
│  server.py
│  work.py
│
└─plugin_config
       ai-plugin.json
       example.yaml
       logo.png
       openapi.yaml
```

插件相关配置文件详细说明可以参考[官方文档](https://agents.baidu.com/docs/develop/ability-plugin/basic/develop_from_scratch/)，以下会做简单说明。

### ai-plugin.json

这个文件描述插件的基本信息，并且包含插件其它配置文件的 URL。

插件基本信息：

- schema_version: 固定为 v1。
- name_for_model: 插件的唯一标识。
- name_for_human: 向用户展示的名字。
- description_for_model: 面向模型的自然语言描述，用于模型参考解析是否触发插件。
- description_for_human: 面向用户介绍插件。
- auth: 需要鉴权功能可以查看[文档](https://agents.baidu.com/docs/develop/ability-plugin/advanced/auth/)，不需要可以直接填写 none。
- contact_email: 安全/审核、支持和停用的电子邮件联系方式。
- legal_info_url: 用户查看插件信息的 URL。

其它配置文件的 URL：

- api: 应该只能为 OpenAPI 类型，并包含 API 描述文件的 URL。
- examples: examples 文件的 URL。
- logo_url: logo 文件的 URL。

URL 中的 "PLUGIN_HOST" 会在服务器代码中被替换为服务器地址。

```json
{
    "schema_version": "v1",
    "name_for_human": "文心悠洛视频分析插件",
    "name_for_model": "wx-yolo-video-inference-plugin",
    "description_for_human": "这是一个视频分析插件，用户可以发送 bilibili 视频链接，获得视频分析的结果。",
    "description_for_model": "这是一个视频分析插件，用户会向你发送 bilibili 视频链接，你需要调用接口获得视频目标检测结果再对视频进行分析。",
    "auth": {
        "type": "none"
    },
    "api": {
        "type": "openapi",
        "url": "PLUGIN_HOST/openapi.yaml"
    },
    "examples": {
        "url": "PLUGIN_HOST/example.yaml"
    },
    "logo_url": "PLUGIN_HOST/logo.png",
    "contact_email": "support@example.com",
    "legal_info_url": "http://www.example.com/legal"
}
```

### openapi.yaml

描述插件为文心智能体提供的接口，格式为 [OpenAPI 标准](https://swagger.io/specification/)。

本项目只需要提供目标检测接口，重点为 `detect_response` 的结构：

- `frames` 完整描述了 `object_detect.py` 返回的目标检测结果的结构。
- `prompt` 字段在 `work.py` 中被添加，以让智能体生成更好的回复。
- `err` 字段会在运行错误时由 `server.py` 返回，让智能体告知用户运行错误的原因。

```yaml
openapi: 3.0.1
info:
    title: wx-yolo
    description: 这是一个视频分析插件，用户会向你发送 bilibili 视频链接，你需要调用接口获得视频目标检测结果再对视频进行分析。
    version: "v1"
servers:
    - url: PLUGIN_HOST
paths:
    /detect:
        post:
            operationId: detect
            summary: 对视频进行目标检测
            requestBody:
                required: true
                content:
                    application/json:
                        schema:
                            $ref: "#/components/schemas/detect_request"
            responses:
                "200":
                    description: 目标检测结果
                    content:
                        application/json:
                            schema:
                                $ref: "#/components/schemas/detect_response"
components:
    schemas:
        detect_request:
            type: object
            required: [url]
            properties:
                url:
                    type: string
                    description: 用户需要分析的视频网址
        detect_response:
            type: object
            description: 目标检测结果
            properties:
                err:
                    type: string
                    description: 如果检测失败，这个字段表示失败的原因
                prompt:
                    type: string
                    description: 提示词
                frames:
                    type: array
                    description: 帧的目标检测结果列表
                    items:
                        type: object
                        description: 目标检测结果
                        properties:
                            time:
                                type: number
                                description: 帧所在视频的时间
                            objs:
                                type: array
                                description: 该帧检测到的目标列表
                                items:
                                    type: object
                                    description: 检测到的目标
                                    properties:
                                        obj:
                                            type: string
                                            description: 目标类型
                                        xywh:
                                            type: array
                                            description: 长度为 4 的整数数组，4 个元素分别代表目标的 X 坐标，Y 坐标，宽度，长度
                                            items:
                                                type: number
```

### example.yaml

编写用户与智能体对话的样例，提升接口调用的正确率。

```yaml
version: 0.0.1
examples:
  - context:
    - role: user
      content: 分析这个奥运会视频 https://www.bilibili.com/video/BV1Yy411i7HY
    - role: bot
      plugin:
        operationId: detect
        thoughts: 用户需要分析视频，给出了URL，现在进行目标检测获取视频内容
        requestArguments:
          url: https://www.bilibili.com/video/BV1Yy411i7HY
```

### 服务器

我们使用 `flask` 框架构建我们的服务器，安装：`pip install flask[async]`

我们要写好智能体获取插件配置文件的接口，这里以 `ai-plugin.json` 为例，其它配置文件和详细上下文可以阅读源码。

```py
@app.route("/ai-plugin.json")
async def plugin_manifest():
    host = request.host_url
    with open("plugin_config/ai-plugin.json", encoding="utf-8") as f:
        text = f.read().replace("PLUGIN_HOST", host)
        return text, 200, {"Content-Type": "application/json"}
```

然后写好目标检测的接口，由于已经把过程封装为一个函数，现在就很简单了：调用 `download_and_detect`，如果出现错误则返回包含 `err` 字段的 JSON，否则返回目标检测结果。

最后还为 result 加入了一个 prompt 字段，以让智能体生成更好的回复。

```py
@app.route("/detect", methods=["POST"])
async def detect_api():
    try:
        frames = await download_and_detect(request.get_json()["url"])
    except Exception as e:
        return make_json_response({"err": "处理失败：" + str(e)})

    res = {
        "frames": frames,
        "prompt": "请不要复述目标检测结果，而是根据结果和用户说明推测一个场景，可以适当修饰。",
    }

    return make_json_response(res)
```

## 效果展示

提供一个链接给智能体：

![成功](./image/dialogue1.png)

提供一个错误的链接给智能体：

![失败](./image/dialogue2.png)

## 项目总结

本文首先介绍了如何使用 PaddleDetect 开发套件进行目标检测，可以体验到 PaddleDetect 部署方便、模型效果优秀。

接着，我们了解了如何将目标检测整合到文心智能体平台，感受到了文心智能体平台的便捷和文心大模型的强大总结能力。
