# from .YOLOv3 import YOLOv3
from .YOLOv3_Sport import YOLOv3_4_SPORT


# __all__ = ['build_detector']
__all__ = ['build_sport_detector']

# def build_detector(cfg, use_cuda):
#     return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES,
#                     score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH,
#                     is_xywh=True, use_cuda=use_cuda)

def build_sport_detector(cfg, use_cuda):
    return YOLOv3_4_SPORT(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES,
                    score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH,
                    is_xywh=True, use_cuda=use_cuda)
