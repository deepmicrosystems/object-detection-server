import cv2
import datetime
import time
import os
import numpy as np

def image_saver(image, PATH_TO_SAVE_IMG):
    unix = int(time.time())
    date = str(datetime.datetime.fromtimestamp(unix).strftime('%Y-%m-%d %H:%M:%S'))
    
    image_path = PATH_TO_SAVE_IMG + f"{date}.jpg"

    print('IMAGE PATH is ' + image_path)
    cv2.imwrite(image_path, image)
    
    return image_path, date

def save_in_db(db, detections, image_path, date):
    finish=False

    db.dynamic_data_entry(image_path=image_path,
                    detection= detections["coord"], 
                    obj_class=detections["class"],
                    prob = detections["probability"],
                    date=date)
    finish = True
    return finish



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
