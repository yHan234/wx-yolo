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
