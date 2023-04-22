from logbook import Logger

from pydantic import BaseModel, ValidationError, validator, Field 
from typing import Union, Optional, List, Dict

logger = Logger(__name__)


class ImagesFromCLients(BaseModel):
    img_data_str : str = None
    class Config: 
        schema_extra = {
                'example': {
                    'image': ('iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAACXBIWXMAAC4jAAAuIwF4pT92AAAA'
                                'B3RJTUUH5AwZAyMzqt+uDgAAABVJREFUGNNj/P//PwNuwMSAF4xUaQCl4wMR/9A5uQAAAABJRU5E'
                                'rkJggg==')
                }
            }
    #@validator("img_data_str")
    #def decode_base64_str(cls, v):
        #try:
            #return base64.b64decode(v)
        #except binascii.Error as e:
            #logger.debug('Received content: {}', v[:12])
            #raise ValueError(str(e)


class Polygon_Plate_Response(BaseModel):
    plate_index : int = None
    plate_type : str = None
    plate_polygon_coords : Dict[str, List[int]]
    plate_number : str = None