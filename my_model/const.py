import numpy as np




IMAGE_SIZE = (320, 320)
EPSILON = 1e-10

LAMBDA_CONFI = 4
LAMBDA_LOC = 1

# CONFIG
HEIGHT, WIDTH = 500, 500
output_shape = [5]  # confi, x, y, w, h




EPOCHS = 10
BATCH_SIZE = 10

# train_steps = len(train_label) // batch_size
# val_steps = len(val_label) // batch_size