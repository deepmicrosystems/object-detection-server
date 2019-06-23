import cv2
import datetime
import time
import os


def image_saver(image, PATH_TO_SAVE_IMG):
    unix = int(time.time())
    date = str(datetime.datetime.fromtimestamp(unix).strftime('%Y-%m-%d %H:%M:%S'))
    
    image_path = PATH_TO_SAVE_IMG + f"{date}.jpg"

    print('IMAGE PATH is ' + image_path)
    cv2.imwrite(image_path, image)
    
    return image_path, date

def save_in_db(db, detections, image_path, date):
    finish=False
    for d in detections["predictions"]:
        db.dynamic_data_entry(image_path=image_path,
                        detection= d["coord"], 
                        obj_class=d["class"],
                        prob = d["probability"],
                        date=date)
        finish = True
    return finish