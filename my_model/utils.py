from cProfile import label
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import glob
from pathlib import Path
from IPython.display import display


def get_label(path, amount=10):
    i=0
    ids = []
    datas = []
    for p in Path(path).iterdir():
        file = p.name
        id = file[:file.index('.')]
        locations = []
        with open(path + file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line[:line.index('\n')]
                bbox_loc = line.split(' ')
                locations.append(bbox_loc)
        
        ids.append(id)
        datas.append(locations)
        
        i+=1
        if i>=amount:
            break
    
    print('Done read_data_label')
    return np.asarray(ids), np.asarray(datas).astype(np.float64)

def get_images(path, ids, size = (320, 320)):
    images = []
    for id in ids:
        img = cv2.imread(path + id + ".jpg", cv2.IMREAD_COLOR)

        # resize
        img = cv2.resize(src=img, dsize=size, interpolation=cv2.INTER_LINEAR)

        # Normalize
        img = np.asarray(img)
        img = img.astype(np.float64)
        img /= 255.0

        images.append(img)
    
    images = np.asarray(images, dtype='object')
    return images

def get_data():
    pos_label = "../../PASCAL_VOC/1-human-label-pos/"
    neg_label = "../../PASCAL_VOC/1-human-label-neg/"
    pos_img = "../../PASCAL_VOC/1-human-images-pos/"
    neg_img = "../../PASCAL_VOC/1-human-images-neg/"

    # GET LABEL
    ids_pos, data_pos = get_label(pos_label)
    ids_neg, data_neg = get_label(neg_label)
    
    data_pos = [data[0] for data in data_pos]
    data_neg = [data[0] for data in data_neg]
    dataset = np.concatenate((data_pos, data_neg), axis=0)

    # GET IMAGES
    img_pos = get_images(pos_img, ids_pos)
    img_neg = get_images(neg_img, ids_neg)
    img_set = np.concatenate((img_pos, img_neg), axis=0)

    return dataset, img_set


if __name__ == '__main__':
    
    data, img = get_data()
    print(data.shape)
    print(img.shape)

    
    print('done utils.py')