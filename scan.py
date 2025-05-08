import cv2
import numpy as np
import tensorflow as tf

def scan(image):
    input_frame = image[:, 280:1000]
    input_frame = cv2.resize(input_frame, (416, 416))
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)