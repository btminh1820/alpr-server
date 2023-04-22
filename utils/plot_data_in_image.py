import cv2
import numpy as np

tst_data = [{'plate_index': 1, 'plate_type': 'short_plate', 'plate_polygon_coords': {'p1': [293, 676], 'p2': [418, 644], 'p3': [437, 753], 'p4': [308, 782]}, 'plate_number': '79-N1/458.35'},
            {'plate_index': 2, 'plate_type': 'short_plate', 'plate_polygon_coords': {'p1': [568, 676], 'p2': [630, 644], 'p3': [650, 753], 'p4': [520, 782]}, 'plate_number': '59-S3/596.51'}]

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def plot_data_to_image(img, data_dict_list):

    img = np.ascontiguousarray(img, dtype=np.uint8)
    h_img, w_img = img.shape[:2]
    for plate_data in data_dict_list:
        plate_poly_points = []
        for points in list(plate_data['plate_polygon_coords'].values()):
            plate_poly_points.append(points)
        pts = np.array(plate_poly_points, np.int32)
        pts_list = pts.tolist()
        # print(type(pts[0][1]))
        pts = pts.reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts],
                            isClosed = True, color = (255, 0, 255), thickness = 5)
        
        ## put license number on image
        label = str(plate_data['plate_index']) + '|' + plate_data['plate_number']
        x_t, y_t = sorted(pts_list, key=lambda elem:elem[0])[0][0], sorted(pts_list, key=lambda elem:elem[1])[0][1]
        if w_img <= 640:
            fontscale = 0.6
        else: 
            fontscale = 1.1

        (w_t, h_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontscale, 2)
        img = cv2.rectangle(img, (x_t, y_t - h_t - 10), (x_t + w_t, y_t), color=(255,0,0),thickness=-1)
        cv2.putText(img=img, text=label,org=(x_t,y_t-5), 
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fontscale, color=(10,0,0), thickness=2)
        
    img_resize = image_resize(img, height=640) 
    #print(img_resize.shape)       
    return img_resize


if __name__ == "__main__":
    tst_img = cv2.imread(r'C:\Users\admin\Downloads\test2.jpg')
    res_img = plot_data_to_image(tst_img, tst_data)    
    #print(res_img.shape)   
    cv2.imshow('res', res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

