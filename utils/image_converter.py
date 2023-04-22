import base64
from io import BytesIO
from tkinter import Image
import numpy as np
from PIL import Image

def img_str_to_np_array(img_base64_string):
    img_bytes = base64.b64decode(img_base64_string)
    img_bytesIO = BytesIO(img_bytes)
    img_bytesIO.seek(0)
    image = Image.open(img_bytesIO)
    img_np_arr = np.array(image)
    img_np_arr = img_np_arr[:,:,::-1]
    return img_np_arr
