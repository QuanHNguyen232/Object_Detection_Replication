import glob
from pathlib import Path
import shutil
import cv2
import pandas as pd
import tensorflow as tf
import numpy as np
from IPython.display import display
import const
from utils import preprocessImage

# DATA FILTERS

def read_file(orig_path, filename):
    line_toadd = []
    with open(orig_path + filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[:line.index(' ', 0)] == '14':
                line = line.replace('14', '1.0', 1)   # label human confidence = 1 (only replace the first '14')
                line_toadd.append(line)
    
    return line_toadd

def filter_human_label(orig_path, target_path):
    # loop through each file
    val=0
    for p in Path(orig_path).glob('*.txt'):
        filename = p.name
        # get all lines in each file
        line_toadd = read_file(orig_path, filename)
        # if human exist in that file
        if len(line_toadd)>0:
            print(f'{filename}: {line_toadd}\n')
            write_to_file(target_path, filename, line_toadd)
        
    print('Done filter_human_label')

def copy_img(file_path, orig_path, target_path):
    val=0
    for p in Path(file_path).iterdir():
        file = p.name[:p.name.index('.')]
        file = file + '.jpg'
        shutil.copy(orig_path + file, target_path)
    print('Done copy_img')

def write_to_file(target_path, file, file_toadd):
    with open(target_path + file, 'w') as f:
        for line in file_toadd:
            f.write(line)

def read_non_human_file(orig_path, filename):
    line_toadd = []
    with open(orig_path + filename, 'r') as f:
        lines = f.readlines()
        isHuman = False
        for line in lines:
            if line[:line.index(' ', 0)] == '14':
                isHuman = True
        # if human not in image, add
        if (not isHuman):
            line_toadd.append('0.0 0.0 0.0 0.0 0.0\n')
    return line_toadd

def filter_non_human_label(orig_path, target_path):
    # loop through each file
    val=0
    for p in Path(orig_path).glob('*.txt'):
        filename = p.name
        line_toadd = read_non_human_file(orig_path, filename)
        if len(line_toadd)>0:
            print(f'{filename}: {line_toadd}\n')
            write_to_file(target_path, file=filename, file_toadd=line_toadd)
            val+=1
        if val==8102:
            break        
    print('Done filter_non_human_label')

def copy_file(file_path, orig_path, target_path):
    count=0
    for p in Path(file_path).iterdir():
        filename = p.name[:p.name.index('.')]
        filename = filename + '.txt'

        with open(orig_path + filename, 'r') as f:
            lines = f.readlines()
            if len(lines)==1:
                shutil.copy(orig_path + filename, target_path)
                count+=1
    print(f'count: {count}')
    print('Done copy_file')


# GET DATA FUNCs
def get_label(path, amount=1000000):
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
                ids.append(id)  # in case there are > 1 person in 1 image (NOT SURE, need to ask)

        datas.append(locations)
        
        i+=1
        if i>=amount:
            break
    
    print('Done read_data_label')
    return np.asarray(ids), np.asarray(datas).astype(np.float64)

def get_images(path, ids):
    images = []
    for id in ids:
        img = cv2.imread(path + id + ".jpg", cv2.IMREAD_COLOR)
        img = preprocessImage(img=img)
        images.append(img)
    
    images = np.asarray(images, dtype='object')
    return images

def get_data(amount=20):
    pos_label = "../../PASCAL_VOC/1-human-label-pos/"
    neg_label = "../../PASCAL_VOC/1-human-label-neg/"
    pos_img = "../../PASCAL_VOC/1-human-images-pos/"
    neg_img = "../../PASCAL_VOC/1-human-images-neg/"

    # GET LABEL
    ids_pos, data_pos = get_label(pos_label, amount=amount)
    ids_neg, data_neg = get_label(neg_label, amount=amount)
    
    data_pos = [data[0] for data in data_pos]
    data_neg = [data[0] for data in data_neg]
    dataset = np.concatenate((data_pos, data_neg), axis=0)

    # GET IMAGES
    img_pos = get_images(pos_img, ids_pos)
    img_neg = get_images(neg_img, ids_neg)
    img_set = np.concatenate((img_pos, img_neg), axis=0)

    return img_set, dataset

if __name__ == '__main__':
    PATH = '../../PASCAL_VOC/'
    file = PATH + 'human-label-neg/'
    img = PATH + 'images/'
    tar = PATH + 'human-images-neg/'

    # FILTER HUMAN IMAGES
    # filter_non_human_label(PATH + 'labels/', PATH + 'human-label-neg/')
    # copy_img(file, img, tar)

    # COPY TO 1 HUMAN
    # copy_file("../../PASCAL_VOC/human-label-pos/", "../../PASCAL_VOC/human-label-pos/", "../../PASCAL_VOC/1-human-label-pos")
    # copy_file("../../PASCAL_VOC/human-label-neg/", "../../PASCAL_VOC/human-label-neg/", "../../PASCAL_VOC/1-human-label-neg/")
    # copy_img("../../PASCAL_VOC/1-human-label-pos/", "../../PASCAL_VOC/human-images-pos/", "../../PASCAL_VOC/1-human-images-pos/")
    # copy_img("../../PASCAL_VOC/1-human-label-neg/", "../../PASCAL_VOC/human-images-neg/", "../../PASCAL_VOC/1-human-images-neg/")

    # READ FILE
    # data = read_label(PATH + 'human-label-neg/')
    # print(data[0])

    # GET DATA
    # data, img = get_data()
    # print(data.shape)
    # print(img.shape)

    print("Done filter_img")
