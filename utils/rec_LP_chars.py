import cv2
from PIL import Image


def lp_chars_rec(model, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    s = model.predict(im_pil)
    return s