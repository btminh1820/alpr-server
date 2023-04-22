import random
import time
from pathlib import Path
import os
import sys

import torch
import torch.backends.cudnn as cudnn

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from Models_Architecture_Source.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, \
    polygon_non_max_suppression, polygon_scale_coords
from Models_Architecture_Source.utils.plots import colors, plot_one_box, polygon_plot_one_box
from Models_Architecture_Source.utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.preprocess_image_yolo import *


def detect_LP_Polygon_in_ROI(model, device, stride, img, imgsz, nms_conf, nms_iou, nms_agnostic, nms_maxdet):
    LP_polygons = [] 

    ROI_image = preprocess_image_yolo_detect(img, device, imgsz, stride)   

    if device != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t0 = time.time()

    # inference
    t1 = time_synchronized()
    with torch.no_grad():
        pred = model(ROI_image)[0]
        #print(pred)

    # Apply polygon NMS
    pred = polygon_non_max_suppression(pred, nms_conf, nms_iou, agnostic=nms_agnostic, max_det=nms_maxdet)
    t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image

        gn = torch.tensor(img.shape)[[1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain xyxyxyxy

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :8] = polygon_scale_coords(ROI_image.shape[2:], det[:, :8], img.shape).round()

            # Write results
            for *xyxyxyxy, conf, cls in reversed(det):
                plate_data = {}
                polygon_int_coords = [int(m.cpu().numpy()) for m in xyxyxyxy]
                class_name = 'long_plate' if int(cls) == 0 else 'short_plate'
                plate_data[class_name] = polygon_int_coords
                LP_polygons.append(plate_data)

    return LP_polygons


# if __name__ == "__main__":
#     yolov5_path = r"D:\license_plate_rec\code\ALPR_server\ALPR_Saved_model\yolov5_Polygon_LP_Det\yolov5s_480_v1_state_dict.pt"
#     model = V5_Model(r'D:\license_plate_rec\code\ALPR_server\ALPR_Saved_model\yolov5_Polygon_LP_Det\yolov5s.yaml')
#     # tst_model = torch.load(yolov5_path, 'cuda:0')
#     # print(tst_model.keys())
#     #model.eval()
#     model.load_state_dict(torch.load(yolov5_path), strict=False)
#     model.to(torch.device('cuda:0'))
#     model.eval()
#     #print(next(model.parameters()).is_cuda)
#     # print('done init yolov5')
#     names = model.module.names if hasattr(model, 'module') else model.names
#     print(names)