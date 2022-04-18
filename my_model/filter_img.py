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

def copy_img(file_path, img_path, target_path):
    val=0
    for p in Path(file_path).iterdir():
        file = p.name[:p.name.index('.')]
        file = file + '.jpg'
        shutil.copy(img_path + file, target_path)
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


def read_label(path):
    i=0
    label = []
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
        
        info = {'id':id, 'data': locations}
        label.append(info)
        i+=1
        if i>=50:
            break
    print('Done read_data_label')
    return label

def get_images(path):
    images = []
    val=0
    for p in Path(path).iterdir():
        file_name = p.name
        images.append(cv2.imread(path + '/' + file_name, cv2.IMREAD_COLOR))
        val+=1
        if val>=50:
            break

    return images

if __name__ == '__main__':
    PATH = '../../PASCAL_VOC/'
    file = PATH + 'human-label-neg/'
    img = PATH + 'images/'
    tar = PATH + 'human-images-neg/'
    # filter_non_human_label(PATH + 'labels/', PATH + 'human-label-neg/')
    # copy_img(file, img, tar)
    
    data = read_label(PATH + 'human-label-neg/')
    print(data[0])

    print("Done filter_img")
