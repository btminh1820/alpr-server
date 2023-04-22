import os
import sys 
from pathlib import Path

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from Models_Architecture_Source.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from Models_Architecture_Source.utils.plots import plot_one_box
from Models_Architecture_Source.utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.preprocess_image_yolo import *


def detect_LP_ROI(model, device, stride, img, imgsz, nms_conf, nms_iou, nms_agnostic):
    result_list = []

    image_processed = preprocess_image_yolo_detect(img, device, imgsz, stride)
    
    if device != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # inference
    t1 = time_synchronized()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(image_processed)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, nms_conf, nms_iou, agnostic=nms_agnostic)
    t3 = time_synchronized()

    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(image_processed.shape[2:], det[:, :4], img.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            res_points = [int(m.cpu().numpy()) for m in xyxy]
            result_list.append(res_points)  # return list of 2d bbox | true image size

    return result_list    



# if __name__ == "__main__":

#     yolo7_path = r"D:\license_plate_rec\code\ALPR_server\ALPR_Saved_model\yolov7_ROI_LP_Det\fin_v1.pt"
#     ckpt = torch.load(yolo7_path, 'cuda:0')
#     print('ckpt keys is', ckpt.keys())
#     tst_model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())
#     model = tst_model[-1]
#     # for param in tst_model.parameters():
#     #     print(param.data)
#     #tst_model.eval()   

#     tst_img = cv2.imread(r'D:\license_plate_rec\dataset\v1\crop_images\ROI_plate\images\val\bike_barrier_01159.jpg')
#     tst_img_process = preprocess_image_yolo_detect(tst_img, 'cuda:0', 640, 32)
#     # # print('tst img is:', tst_img_process)
#     with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
#             pred = model(tst_img_process)[0]
#     pred = non_max_suppression(pred, 0.2, 0.5, False)
#     print('pred after nms', pred)
    # for i, det in enumerate(pred):  # detections per image
    #     gn = torch.tensor(tst_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #     if len(det):
    #         # Rescale boxes from img_size to im0 size
    #         det[:, :4] = scale_coords(tst_img_process.shape[2:], det[:, :4], tst_img.shape).round()
    #         #print(det)
    #     for *xyxy, conf, cls in reversed(det):
    #         print('detected xyxy', xyxy)    
    #         #result_list.append(xyxy)  # return list of 2d bbox | true image size    