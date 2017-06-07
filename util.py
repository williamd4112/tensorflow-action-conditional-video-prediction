import cv2
import numpy as np

def post_process(x, mean):
    x *= 255.0
    x += mean
    x = x.astype(np.uint8)
    x = np.clip(x, 0, 255)
    return x
