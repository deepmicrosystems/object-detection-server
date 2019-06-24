import cv2
import datetime
import time
import os
import numpy as np


def object_detection(image_np, min_score, draw_box, detect):

    detection = None
    detection_with_filter = None

    detection = detect.detect(image_np)

    boxes = detection['boxes']
    scores = detection['scores']
    classes = detection['classes']
    num = detection['num']
    
    detection_with_filter, frame = detect.annotate_image_and_filter(image_np,
                                            boxes, 
                                            classes, 
                                            scores, 
                                            num, 
                                            min_score, 
                                            draw_box)

    return detection_with_filter, frame

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)



def prepare_image(image):
    image = np.asarray(image)
    return image
