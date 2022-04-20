import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import glob
from pathlib import Path
from IPython.display import display
import const

##################################################
#          INTERSECTION OVER UNION               #
##################################################
def basicIOU(boxes1, boxes2):
    bboxes1 = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0, # x - (w/2) = x1
                         boxes1[..., 1] - boxes1[..., 3] / 2.0, # y - (h/2) = y1
                         boxes1[..., 0] + boxes1[..., 2] / 2.0, # x + (h/2) = x2
                         boxes1[..., 1] + boxes1[..., 3] / 2.0],    # y + (h/2) = y2
                        axis=-1)

    bboxes2 = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                         boxes2[..., 1] - boxes2[..., 3] / 2.0,
                         boxes2[..., 0] + boxes2[..., 2] / 2.0,
                         boxes2[..., 1] + boxes2[..., 3] / 2.0],
                        axis=-1)

    # calculate the left up point & right down point
    lu = tf.maximum(bboxes1[..., :2], bboxes2[..., :2])
    rd = tf.minimum(bboxes1[..., 2:], bboxes2[..., 2:])

    # intersection
    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]

    # calculate the boxs1 square and boxs2 square
    square1 = boxes1[..., 2] * boxes1[..., 3]
    square2 = boxes2[..., 2] * boxes2[..., 3]

    union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0) 

def multiBoxIOU(boxes1, boxes2):
    boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                         boxes1[..., 1] - boxes1[..., 3] / 2.0,
                         boxes1[..., 0] + boxes1[..., 2] / 2.0,
                         boxes1[..., 1] + boxes1[..., 3] / 2.0],
                        axis=-1)

    boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                         boxes2[..., 1] - boxes2[..., 3] / 2.0,
                         boxes2[..., 0] + boxes2[..., 2] / 2.0,
                         boxes2[..., 1] + boxes2[..., 3] / 2.0],
                        axis=-1)

    # calculate the left up point & right down point
    lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
    rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

    # intersection
    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]

    # calculate the boxs1 square and boxs2 square
    square1 = boxes1[..., 2] * boxes1[..., 3]
    square2 = boxes2[..., 2] * boxes2[..., 3]

    union_square = tf.maximum(square1 + square2 - inter_square, const.EPSILON)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


##################################################
#               PRE-PROCESS IMAGE                #
##################################################
def preprocessImage(img, image_size=const.IMAGE_SIZE):
     # resize
    img = cv2.resize(src=img, dsize=image_size, interpolation=cv2.INTER_LINEAR)

    # Normalize
    img = np.asarray(img)
    img = img.astype(np.float64)
    img /= 255.0

    return img



##################################################
#               IMAGE DISPLAY                    #
##################################################
def open_img(file_path=None, x=0, y=0, w=10, h=10, label='image'):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    
    add_bbox(img, start_X=x, start_Y=y, rec_w=w, rec_h=h, label=label)
    cv2.imshow(file_path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_webcam(file_path=0, mirror=False, label='webcam/mp4'):
    cam = cv2.VideoCapture(file_path, cv2.CAP_DSHOW)

    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)

        add_bbox(img, 150, 150, 200, 100, label)
        cv2.imshow('from webcam/mp4', img)
        # time.sleep(1)
        if cv2.waitKey(1) > -1: # press any key. Ascii table: 0 is smallest index
            break
    cv2.destroyAllWindows()

def add_bbox(img, start_X, start_Y, rec_w, rec_h, label=None):
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    box_color = (0, 0, 255)

    # print text + background
    img = cv2.rectangle(img, (start_X, start_Y - 20), (start_X + text_w, start_Y), box_color, -1)
    img = cv2.putText(img, label, (start_X, start_Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    # print border
    img = cv2.rectangle(img, (start_X, start_Y), (start_X + rec_w, start_Y + rec_h), box_color, 2)

    return img

def testShowImage(isWebCam=False, file_path=None, x=0, y=0, w=0, h=0):
    if isWebCam:
        show_webcam(file_path=file_path, mirror=True)
    else:
        open_img(file_path, x=x, y=y, w=w, h=h)
    

if __name__ == '__main__':
    
    

    

    print('done utils.py')