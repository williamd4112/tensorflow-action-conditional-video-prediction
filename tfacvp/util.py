import cv2
import numpy as np

def post_process(x, mean, scale=255.0):
    x *= scale
    x += mean
    x = np.clip(x, 0, scale)
    x = x.astype(np.uint8)
    return x
