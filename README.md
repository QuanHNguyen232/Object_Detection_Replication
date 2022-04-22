# Real-time_human-detection

<!-- Based on [https://github.com/Ghulamrasool11/Human-Detection-with-Convolution-neural-network-](https://github.com/Ghulamrasool11/Human-Detection-with-Convolution-neural-network-) -->

<!-- dataset:
    * untrained [https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset)
    * trained [https://www.crowdhuman.org/](https://www.crowdhuman.org/) -->


<!-- source: [not use CNN](https://data-flair.training/blogs/python-project-real-time-human-detection-counting/) -->

I have 2 models in this repo. 

* The first one is my-model, which I am trying to build my self from scratch.
    * `const.py`: has constants such as EPSILON, LAMBDA in loss function, IMAGE_SIZE, etc.
    * `dataset.py`: including functions to load data (images, labels), filter out images.0
    * `loss.py`: compute loss for model
    * `main.py`: main file
    * `model.py`: has YOLOv1 model using Conv2D and Pooling layers
    * `utils.py`: contains functions such as read images, webcam, mp4 files, or process image, and calculate Intersection Over Union (IOU)

* YOLOv3 from scratch: [Youtube](https://www.youtube.com/watch?v=Grir6TZbc1M) using [config file](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) from the author

Dataset:
* I use [PASCAL_VOC](https://www.kaggle.com/datasets/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video) dataset.
* For my-model, since it is only for 1 person each image, we must use filter functions to filter out images with human-only labels.
