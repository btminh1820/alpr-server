import json
import io 
import os
import numpy as np
import cv2

import requests
import streamlit as st
import base64
from PIL import Image
from io import BytesIO
from pyngrok import ngrok, conf


from utils.plot_data_in_image import plot_data_to_image


headers = {'content-type': 'application/json'}

st.set_option("deprecation.showfileUploaderEncoding", False)
st.title("License Plate API")

image = st.file_uploader("Choose an image", type=['jpg', 'jpeg'])
#imageLocation = st.empty()

if image:
    st.image(image, caption = "Your upload image")

# test_options = st.selectbox("Choose testing utils", [i for i in TESTING_UTILS_LIST.values()])    

if st.button("Start Detect and Rec"):
    if image is not None: 
        img_data_bytes = image.getvalue()
        file_bytes =  np.asarray(bytearray(image.read()), dtype=np.uint8)
        #file_bytes = np.ascontiguousarray(bytearray(image.read()), dtype=np.uint8)
        img_np = cv2.imdecode(file_bytes, 1)
        img_np = img_np[:,:,::-1]
        img_base64_bytes = base64.b64encode(img_data_bytes)
        img_base64_str = img_base64_bytes.decode("ascii")

        file_upload = json.dumps({"img_data_str": img_base64_str})
        res = requests.post(f"http://127.0.0.1:8000/start/polygon_plates_rec/", file_upload, headers=headers)
        #print('res is', res)
        res_dict = res.json()  

        ### display res image and res json
        st.info(body=' VIEW PREDICTION BELOW ⬇️')
        res_image =  plot_data_to_image(img_np, res_dict)
        st.image(res_image, caption='Results')   
        st.json(res_dict)


### set up to public front end - using ngrok
### 
# conf.get_default().config_path = r'C:\Users\admin\.ngrok2\ngrok.yml'
# public_url = ngrok.connect(port = '80')
# print(f"Please click on the text below {public_url}")
