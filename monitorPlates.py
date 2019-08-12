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

import libs.ssd.ssd_processor.models as Models


# Recognize the following arguments to control the program flow
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",      type = str,  default = None,       help = "Set stream input")
parser.add_argument("-o", "--output",     type = str,  default = "./output", help = "Set images output")
parser.add_argument("-s", "--show",       type = bool, default = False,      help = "Do you wish to show image?")
parser.add_argument("-p", "--proc_scale", type = int,  default = 5,          help = "Set processing factor")
parser.add_argument("-kt", "--kitty",     type = bool, default = False,      help = "Use experimental Kitty model")
parser.add_argument("-k", "--kill",       type = int,  default = 0,          help = "Set self kill time")
args = parser.parse_args()

if args.kitty:
    print("Using Kitty Model")
    kitti_model = Models.KittiModel()
else:
    print("Using Coco Model")
    kitti_model = Models.MSCOCOModel()

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
    auxiliar_time = start_time

    while True:
        ret, frame = miCamara.read()
        if frame.shape[2] > 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        processing_frame = cv2.resize(frame,(frame.shape[1]//PROCESS_SCALE,frame.shape[0]//PROCESS_SCALE))
        
        #detections = detect.detect(frame)
        detections,drawing_frame = detect.object_detection(processing_frame,0.5,True)

        if args.show:
            #cv2.imshow("Monitor",cv2.resize(frame,(320,240)))
            cv2.imshow("Drawing",drawing_frame)

        print(drawing_frame.shape)
        print(detections)
        str_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("Current time: ",str_datetime)
        
        for current_detection in detections["predictions"]:
            
            detected_object = current_detection["class"]
            print("Found {}".format(detected_object))
            if detected_object in ["car","bus","truck"]:
                print("Original shape: ",frame.shape)
                y_min = PROCESS_SCALE*current_detection["coord"]["ymax"]
                y_max = PROCESS_SCALE*current_detection["coord"]["ymin"]
                x_min = PROCESS_SCALE*current_detection["coord"]["xmax"]
                x_max = PROCESS_SCALE*current_detection["coord"]["xmin"]
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

        print("Period time: {:0.2f}".format(time()-auxiliar_time))
        auxiliar_time = time()

        if (args.kill) and (auxiliar_time-start_time):
            print("Self killing app")
            break

        ch = cv2.waitKey(1)

        if ch == ord("q"):
            break
    