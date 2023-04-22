import os
from os.path import join
from typing import Union

import torch
from pydantic import BaseSettings
from pydantic.main import BaseModel


BASE_PATH_CONFIG = os.path.dirname(__file__)

class YOLOV7_ROI_MODEL(BaseSettings):
    WEIGHTS: str =  os.path.join(BASE_PATH_CONFIG, r"ALPR_Saved_model\yolov7_ROI_LP_Det\v1.pt")
    CONF: float = 0.25
    STRIDE_MAX : int = 32
    IMAGE_SIZE : int = 640
    NMS_CONF: float = 0.25
    NMS_IOU_THRESH: float = 0.5
    NMS_AGNOSTIC: bool = False


class YOLOV5_POLYGON_MODEL(BaseSettings):
    WEIGHTS: str =  os.path.join(BASE_PATH_CONFIG, r"ALPR_Saved_model\yolov5_Polygon_LP_Det\yolov5m_480_v1.pt")
    CONF: float = 0.3
    STRIDE_MAX: int = 32
    IMAGE_SIZE : int = 480
    NMS_CONF: float = 0.8
    NMS_IOU_THRESH: float = 0.5
    NMS_AGNOSTIC: bool = False
    NMS_MAX_DET: int = 1000


class VIETOCR_LP_CHARACTERS_MODEL(BaseSettings):
    CONFIG_FILE: str =  os.path.join(BASE_PATH_CONFIG, r"ALPR_Saved_model\VietOCR_LP_Rec\config_file\vietocr_lp_rec_v1.yaml")
    VOCAB : str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.'
    WEIGHTS : str = os.path.join(BASE_PATH_CONFIG, r"ALPR_Saved_model\VietOCR_LP_Rec\model\vietocr_lp_rec_v1.pth")
    #VIETOCR_CONFIG_DICT: dict = vietocr_config


class Settings(BaseSettings):
    YOLOV7_ROI: Union[YOLOV7_ROI_MODEL, None] = YOLOV7_ROI_MODEL()
    YOLOV5_POLYGON: Union[YOLOV5_POLYGON_MODEL, None] = YOLOV5_POLYGON_MODEL()
    VIETOCR_REC: Union[VIETOCR_LP_CHARACTERS_MODEL, None] = VIETOCR_LP_CHARACTERS_MODEL()
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    class Config:
        env_file = '.env'


settings = Settings()

