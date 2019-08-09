"""
How to use it
$ curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
"""

import numpy as np
from flask import Flask, render_template, request, jsonify, Response
import io
from PIL import Image
import json
import os
import time
import datetime

from libs.ssd.ssd_processor import SSDProcessor
from libs.ssd.ssd_processor.models import KittiModel

from libs.tools import *

from libs.db.db_handler import DataBaseManager
from libs.image_processor import ImageProcessor

kitti_model = KittiModel()
MIN_SCORE_THRESH = 0.6
DRAW_BOX = True

PATH_TO_SAVE_IMG = os.path.join(os.path.dirname(__file__), "images/")
detect = SSDProcessor(model = kitti_model)
detect.setup()

# DB
my_db = DataBaseManager()


# Image processor

imageProcessor = ImageProcessor()
# Flask App Globals
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    global detect
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):
            image_path_request = request.files["image"]
            print('image path request', image_path_request)
            # read the image in PIL format
            image = image_path_request.read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image_np = prepare_image(image)
            image_shape = image_np.shape
            # TODO
            # handle /endpoint for plate recognition
            # separate /detection endpoint 
            # separete db logic for plates and detections
            detection_with_filter, image_np_lr_draw = object_detection(image_np_lr, MIN_SCORE_THRESH, DRAW_BOX, detect)
            
            if detection_with_filter['success']:
                db_path = os.path.join(os.path.dirname(__file__),   
                                        "libs", "db", "test_sqlite.db" )

                db_started = imageProcessor.start_modules(my_db, db_path)

                state = imageProcessor(db_path, detection_with_filter, image_np_lr_draw, path_to_img, item_id)
                
                print('SAVED?', done)
            else:
                detection_with_filter['success'] = False
                
            resp = Response(response=json.dumps(detection_with_filter),
                                status=200,
                                mimetype="application/json")
            return resp



@app.route("/predict_video", methods=["POST"])
def predict_video():
    global detect
    # initialize the data dictionary that will be returned from the
    # view
    detection_with_filter = {}
    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        nparr = np.fromstring(request.data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)            

        # preprocess the image and prepare it for classification
        image = prepare_image(img_np)
        
        detection_with_filter, frame = object_detection(image, MIN_SCORE_THRESH, DRAW_BOX, detect)
        
        if len(detection_with_filter["predictions"]) > 0:
            db_path = os.path.join(os.path.dirname(__file__),   
                                    "libs", "db", "test_sqlite.db" )

            detection_with_filter['success'] = True
        else:
            detection_with_filter['success'] = False
            
        resp = Response(response=json.dumps(detection_with_filter),
                            status=200,
                            mimetype="application/json")
        return resp


if __name__ == '__main__':

    detect.setup()
    time.sleep(15)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

