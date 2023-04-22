import sys
import os

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from config import settings 
from Models_Architecture_Source.models.experimental import attempt_load
from Models_Architecture_Source.vietocr_root.vietocr.tool.predictor import Predictor
from Models_Architecture_Source.vietocr_root.vietocr.tool.config import Cfg


def init_yolov7(yolov7_weights):
    model = attempt_load(yolov7_weights, settings.DEVICE)
    return model

def init_yolov5_polygon(yolov5_polygon_weights):
    model = attempt_load(yolov5_polygon_weights, settings.DEVICE)
    return model    

def init_vietocr(vietocr_config_path):
    config = Cfg.load_config_from_file(vietocr_config_path)
    config['vocab'] = settings.VIETOCR_REC.VOCAB
    config['weights'] = settings.VIETOCR_REC.WEIGHTS
    config['device'] = settings.DEVICE
    detector = Predictor(config)
    return detector


ROI_Det_Model = init_yolov7(settings.YOLOV7_ROI.WEIGHTS)
print('Done init YOLOv7_ROI_Model')

POLYGON_Det_Model = init_yolov5_polygon(settings.YOLOV5_POLYGON.WEIGHTS)
print('Done init Yolov5_Polygon_Model')

CHARS_Rec_Model = init_vietocr(settings.VIETOCR_REC.CONFIG_FILE)
print('Done init VietOCR Model')
