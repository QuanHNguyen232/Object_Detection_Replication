import glob
from pathlib import Path
import shutil
import cv2





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

    print("Done filter_img")
