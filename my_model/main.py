import cv2

def open_img(file_path=None):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    start_point = (50, 500) # top left corner of rectangle
    end_point = (220, 220)  # bottom right corner of rectangle
    color = (0, 0, 255)
    thickness = 2   # border thickness
    image = cv2.rectangle(img, start_point, end_point, color, thickness)
    cv2.imshow(file_path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_webcam(file_path=0, mirror=False):
    cam = cv2.VideoCapture(file_path)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        add_rec(img, rec_w=50, rec_h=100)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) > -1: # press any key. ESC=27 (ascii table)
            break
    cv2.destroyAllWindows()

def add_rec(src, rec_w, rec_h, start_X = 100, start_Y = 100):
    color=(0,0,255)
    src = cv2.rectangle(img=src, pt1=(start_X, start_Y), pt2=(start_X + rec_w, start_Y + rec_h), color=color, thickness=2)
    src = cv2.putText(img=src, text="test", org=(start_X, start_Y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=2)

def main(isWebCam=False, file_path=None):
    if isWebCam:
        show_webcam(file_path=file_path, mirror=False)
    else:
        open_img(file_path)

if __name__ == '__main__':
    main(isWebCam=True, file_path=0)
