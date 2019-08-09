import os
import sys
import cv2
import argparse
from libs.ssd.ssd_processor import SSDProcessor
sys.path.append(os.getenv('HOME')+'/trafficFlow/preview/')
from playstream import PlayStream

from libs.ssd.ssd_processor.models import KittiModel
kitti_model = KittiModel()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type = str, default = None, help = "Set stream input")
args = parser.parse_args()


detect = SSDProcessor(model = kitti_model)
detect.setup()

if __name__ == "__main__":
    miCamara = PlayStream(args.input)


    while True:
        ret, frame = miCamara.read()
        
        #cv2.imshow("Monitor",cv2.resize(frame,(320,240)))
        #detections = detect.detect(frame)
        detections,frame2 = detect.object_detection(frame,0.5,True)
        cv2.imshow("Objects",frame2)
        print(frame2.shape)
        print(detections)

        ch = cv2.waitKey(1)

        if ch == ord("q"):
            break
    