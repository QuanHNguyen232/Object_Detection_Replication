import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import glob
from pathlib import Path
from IPython.display import display
import const

#================================================#
#          INTERSECTION OVER UNION               #
#================================================#
def basicIOU(bboxes1, bboxes2):
    # Get center x, y, get w, h of bbox
    box1_x = bboxes1[..., 0:1]
    box1_y = bboxes1[..., 1:2]
    box1_w = bboxes1[..., 1:3]
    box1_h = bboxes1[..., 3:4]
    
    box2_x = bboxes2[..., 0:1]
    box2_y = bboxes2[..., 1:2]
    box2_w = bboxes2[..., 1:3]
    box2_h = bboxes2[..., 3:4]

    # get corner x, y of bbox
    box1_x1 = box1_x - (box1_w / 2.0)
    box1_y1 = box1_y - (box1_h / 2.0)
    box1_x2 = box1_x + (box1_w / 2.0)
    box1_y2 = box1_y + (box1_h / 2.0)
    print(f'box_1: ({box1_x1}:{box1_y1}) ({box1_x2}:{box1_y2})')

    box2_x1 = box2_x - (box2_w / 2.0)
    box2_y1 = box2_y - (box2_h / 2.0)
    box2_x2 = box2_x + (box2_w / 2.0)
    box2_y2 = box2_y + (box2_h / 2.0)
    print(f'box_2: ({box2_x1}:{box2_y1}) ({box2_x2}:{box2_y2})')

    # get corner of union box
    union_x1 = tf.maximum(box1_x1, box2_x1)
    union_y1 = tf.maximum(box1_y1, box2_y1)
    union_x2 = tf.minimum(box1_x2, box2_x2)
    union_y2 = tf.minimum(box1_y2, box2_y2)
    print(f'union_box: ({union_x1}:{union_y1}) ({union_x2}:{union_y2})')

    # get union box area
    union_w = tf.maximum(0.0, union_x2 - union_x1)
    union_h = tf.maximum(0.0, union_y2 - union_y1)
    union_area = union_w * union_h
    print(f'union_area: {union_area}')

    # get intersect area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box1_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    intersect_area = (box1_area + box1_area) - union_area
    print(f'intersect_area: {intersect_area}')

    # get iou, add EPSILON to prevent num/0
    iou = union_area / (intersect_area + const.EPSILON)

    return iou

def multiBoxIOU(boxes1, boxes2):
    boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                         boxes1[..., 1] - boxes1[..., 3] / 2.0,
                         boxes1[..., 0] + boxes1[..., 2] / 2.0,
                         boxes1[..., 1] + boxes1[..., 3] / 2.0],
                        axis=-1)

    boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                         boxes2[..., 1] - boxes2[..., 3] / 2.0,
                         boxes2[..., 0] + boxes2[..., 2] / 2.0,
            1ate the left up point & right down point
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

def testIOU():
    box1 = tf.convert_to_tensor([[2, 3, 2, 4], [3, 8.5, 2, 3], [2, 3, 2, 4]], dtype=tf.float32)
    box2 = tf.convert_to_tensor([[4, 5, 4, 2], [2, 3, 2, 4]], [6.5, 3.5, 3, 3],dtype=tf.float32)

    print(f'box1_shape: {box1.shape}')
    print(f'box2_shape: {box2.shape}')

    result = basicIOU(box1, box2)
    print(f'result_shape: {result.shape}')
    print(f'result: {result}')
    '''
    expect:
        union area: [1, 0, 0]
        intersect_area: [15, 14, 17]
    '''


#================================================#
#               PRE-PROCESS IMAGE                #
#================================================#
def preprocessImage(img, image_size=const.IMAGE_SIZE):
     # resize
    img = cv2.resize(src=img, dsize=image_size, interpolation=cv2.INTER_LINEAR)

    # Normalize
    img = np.asarray(img)
    img = img.astype(np.float64)
    img /= 255.0

    return img



#================================================#
#               IMAGE DISPLAY                    #
#================================================#
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