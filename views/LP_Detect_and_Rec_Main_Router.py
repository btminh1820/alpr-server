import numpy as np 

from logbook import Logger
from fastapi import Depends, HTTPException, Request, APIRouter, BackgroundTasks
from typing import List

from schema import ImagesFromCLients, Polygon_Plate_Response
from utils.image_converter import img_str_to_np_array
from utils.detect_LP_ROI import detect_LP_ROI
from utils.detect_LP_polygon import detect_LP_Polygon_in_ROI
from utils.rectify_LP_Polygons import polygon_plate_rectify
from utils.rec_LP_chars import lp_chars_rec
from utils.init_models import ROI_Det_Model, POLYGON_Det_Model, CHARS_Rec_Model
from config import settings

logger = Logger(__name__)
router = APIRouter()

@router.post("/start/polygon_plates_rec/", response_model=List[Polygon_Plate_Response])
def polygon_plates_num_detect(img_upload : ImagesFromCLients):
    RES_DATA = []
    plate_index = 0
    img_np = img_str_to_np_array(img_upload.img_data_str)
    
    ## detect Plate ROI - Polygon Plates and assign index
    Plate_Roi_Coords = detect_LP_ROI(model=ROI_Det_Model, device=settings.DEVICE, stride=settings.YOLOV7_ROI.STRIDE_MAX,
                                    img=img_np, 
                                    imgsz=settings.YOLOV7_ROI.IMAGE_SIZE, nms_conf=settings.YOLOV7_ROI.CONF, 
                                    nms_iou=settings.YOLOV7_ROI.NMS_IOU_THRESH, 
                                    nms_agnostic=settings.YOLOV7_ROI.NMS_AGNOSTIC)
    #print(Plate_Roi_Coords)
    for roi_coord in Plate_Roi_Coords:
        roi_image = img_np[roi_coord[1]:roi_coord[3], roi_coord[0]:roi_coord[2]]
        #cv2.imwrite('test_roi_img.jpg', roi_image)
        polygon_plate_coords_type = detect_LP_Polygon_in_ROI(POLYGON_Det_Model, device=settings.DEVICE, 
                                                img=roi_image, stride=settings.YOLOV5_POLYGON.STRIDE_MAX,
                                                imgsz=settings.YOLOV5_POLYGON.IMAGE_SIZE, nms_conf=settings.YOLOV5_POLYGON.NMS_CONF,
                                                nms_iou=settings.YOLOV5_POLYGON.NMS_IOU_THRESH, 
                                                nms_agnostic=settings.YOLOV5_POLYGON.NMS_AGNOSTIC,
                                                nms_maxdet=settings.YOLOV5_POLYGON.NMS_MAX_DET)
        # print('polygon plate coords type are', polygon_plate_coords_type)
        for plate_data in polygon_plate_coords_type:
            recognized_data = {}
            plate_index += 1
            plate_type = list(plate_data.keys())[0]
            plate_polygon_coords = list(plate_data.values())[0]

            # split 1 xyxyxyxy list into 4 list 
            res_plate_polygon_coords = np.array(plate_polygon_coords, dtype=np.int32)
            res_plate_polygon_coords = np.split(res_plate_polygon_coords, 4)
            #print('res_plate_polygon_coords', res_plate_polygon_coords)

            # rectify polygon coords | 
            rect_plate_image = polygon_plate_rectify(roi_image, res_plate_polygon_coords)
            #cv2.imwrite('rectified_image.jpg', rect_plate_image)
            h,w = rect_plate_image.shape[:2]

            # rec character in long plate or short plate
            if plate_type == 'long_plate':
                rec_chars = lp_chars_rec(CHARS_Rec_Model, rect_plate_image) #string
            else:
                height_cutoff = h // 2
                upper_half_plate_img = rect_plate_image[:height_cutoff, :]
                #cv2.imwrite('upper_half_img.jpg', upper_half_plate_img)
                lower_half_plate_img = rect_plate_image[height_cutoff:, :]
                #cv2.imwrite('lower_half_img.jpg', lower_half_plate_img)
                upper_chars = lp_chars_rec(CHARS_Rec_Model, upper_half_plate_img)
                lower_chars = lp_chars_rec(CHARS_Rec_Model, lower_half_plate_img)
                rec_chars = upper_chars + '/' + lower_chars
            #print('rec_chars', rec_chars)
            recognized_data['plate_index'] = int(plate_index)
            recognized_data['plate_type'] = str(plate_type)
            recognized_data['plate_polygon_coords'] = {'p1': [int(res_plate_polygon_coords[0][0] + roi_coord[0]), int(res_plate_polygon_coords[0][1] + roi_coord[1])], 
                                                 'p2': [int(res_plate_polygon_coords[1][0] + roi_coord[0]), int(res_plate_polygon_coords[1][1] + roi_coord[1])],
                                                 'p3': [int(res_plate_polygon_coords[2][0] + roi_coord[0]), int(res_plate_polygon_coords[2][1] + roi_coord[1])],
                                                 'p4': [int(res_plate_polygon_coords[3][0] + roi_coord[0]), int(res_plate_polygon_coords[3][1] + roi_coord[1])]
                                                 }
            recognized_data['plate_number'] = str(rec_chars)
            # print('res_data', recognized_data)
            # print(recognized_data)
            RES_DATA.append(recognized_data)
      
    return RES_DATA

