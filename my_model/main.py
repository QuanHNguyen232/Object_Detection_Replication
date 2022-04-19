import cv2
import time
from dataset import read_data_label

def open_img(file_path=None, x=0, y=0, w=10, h=10):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img_width, img_height, channel = img.shape
    print(f'{img_width} {img_height}')

    start_X=float(x)*img_width
    start_Y=float(y)*img_height
    rec_w=float(w)*img_width
    rec_h=float(h)*img_height

    print(f'start_X={start_X} start_Y={start_Y}, rec_w={rec_w}, rec_h={rec_h}')
    add_bbox(img, start_X=start_X, start_Y=start_Y, rec_w=rec_w, rec_h=rec_h)
    cv2.imshow(file_path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_webcam(file_path=0, mirror=False):
    cam = cv2.VideoCapture(file_path, cv2.CAP_DSHOW)

    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)

        add_bbox(img, rec_w=50, rec_h=100, start_X=150, start_Y=150)
        cv2.imshow('my webcam', img)
        # time.sleep(1)
        if cv2.waitKey(1) > -1: # press any key. ESC=27 (ascii table)
            break
    cv2.destroyAllWindows()

def add_bbox(src, start_X, start_Y, rec_w, rec_h):
    src = cv2.rectangle(img=src, pt1=(400, 100), pt2=(400 + 50, 100 + 200), color=(0,0,255), thickness=2)
    # src = cv2.putText(img=src, text="test", org=(start_X, start_Y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,255,0), thickness=2)

def main(isWebCam=False, file_path=None, x=0, y=0, w=0, h=0):
    if isWebCam:
        show_webcam(file_path=file_path, mirror=True)
    else:
        open_img(file_path, x=x, y=y, w=w, h=h)

if __name__ == '__main__':
    # main(isWebCam=False, file_path='imgs/hiking.jpg')
    data = read_data_label('../../PASCAL_VOC/')
    # print(data)
    # print(data[0]['data'][0])
    class_, x, y, w, h = data[26]['data'][0]
    print(f'{x} {y} {w} {h}')
    open_img(file_path="../../PASCAL_VOC/human-images/000027.jpg", x=x, y=y, w=w, h=h)
    # main(isWebCam=True, file_path=0)