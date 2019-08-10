import os
import sys
import cv2
import json
import argparse
from time import time
from openalpr import Alpr
from datetime import datetime
from libs.ssd.ssd_processor import SSDProcessor
sys.path.append(os.getenv('HOME')+'/trafficFlow/preview/')
from playstream import PlayStream

from libs.ssd.ssd_processor.models import KittiModel
kitti_model = KittiModel()

# Recognize the following arguments to control the program flow
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",      type = str,  default = None,       help = "Set stream input")
parser.add_argument("-o", "--output",     type = str,  default = "./output", help = "Set images output")
parser.add_argument("-s", "--show",       type = bool, default = False,      help = "Do you wish to show image?")
parser.add_argument("-p", "--proc_scale", type = int,  default = 8,         help = "Set processing factor")
args = parser.parse_args()

# Constants
PROCESS_SCALE = args.proc_scale

# Variables
counter = 0
none_plates = 0
good_plates = 0

# Configure the object detector:
detect = SSDProcessor(model = kitti_model)
detect.setup()

# Configure the ALPR detector
alpr = Alpr("us", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data")
#print(dir(alpr))

# Set up the output folder
if not os.path.exists(args.output):
    os.mkdir(args.output)

if __name__ == "__main__":
    miCamara = PlayStream(args.input)
    start_time = time()

    while True:
        ret, frame = miCamara.read()
        processing_frame = cv2.resize(frame,(frame.shape[1]//PROCESS_SCALE,frame.shape[0]//PROCESS_SCALE))
        
        #detections = detect.detect(frame)
        detections,drawing_frame = detect.object_detection(processing_frame,0.5,True)

        if args.show:
            #cv2.imshow("Monitor",cv2.resize(frame,(320,240)))
            cv2.imshow("Drawing",drawing_frame)
        

        print(drawing_frame.shape)
        print(detections)
        if len(detections["predictions"])>0:
            str_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            detected_object = detections["predictions"][0]["class"]
            print("Found {}".format(detected_object))
            print("Original shape: ",frame.shape)
            y_min = PROCESS_SCALE*detections["predictions"][0]["coord"]["ymax"]
            y_max = PROCESS_SCALE*detections["predictions"][0]["coord"]["ymin"]
            x_min = PROCESS_SCALE*detections["predictions"][0]["coord"]["xmax"]
            x_max = PROCESS_SCALE*detections["predictions"][0]["coord"]["xmin"]
            print("Coordinates ", x_min, y_min, " to ", x_max, y_max)
            found_object = frame[y_min:y_max,x_min:x_max]
            if args.show:
                cv2.imshow("Object",found_object)
            print("Object shape: ",found_object.shape)
            results = alpr.recognize_ndarray(found_object)
            print("Results: {}".format(results))

            # Save results to a dictionary:
            print("Saving results:")
            dictionary = {  "object_detection":detections,
                            "openalpr_detection":results}

            base_name = args.output+"/{}_{}_{}_".format(str(counter).zfill(6),str_datetime,detected_object)

            try:
                file_name = base_name + "{}".format(results["results"]["plate"])
                good_plates += 1
            except:
                print("Did not found any information, setting a default name")
                file_name = base_name + "none"
                none_plates += 1
            
            json_name = file_name + ".json"
            
            full_size_image_name = file_name + "_full.jpg"
            object_image_name = file_name + "_object.jpg"

            # Saving image and data:
            cv2.imwrite(full_size_image_name,frame)
            cv2.imwrite(object_image_name, found_object)

            json_data = json.dumps(dictionary)
            print("Writing data {}".format(json_data))
            f = open(file_name,"w")
            f.write(json_data)
            f.close()
            print("Ratio: Good {} Bad: {}".format(good_plates, none_plates))
            counter += 1

        print("Period time: {:0.2f}".format(time()-start_time))
        start_time = time()

        ch = cv2.waitKey(1)

        if ch == ord("q"):
            break
    