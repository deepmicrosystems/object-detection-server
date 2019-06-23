"""
How to use it
$ curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
"""

import numpy as np
from flask import Flask, render_template, request, jsonify, Response
import io
from PIL import Image
import json
import time
from libs.ssd.ssd_processor import SSDProcessor
from libs.ssd.ssd_processor.models import KittiModel


kitti_model = KittiModel()

MIN_SCORE_THRESH = 0.3
DRAW_BOX = True


detect = SSDProcessor(model = kitti_model)
detect.setup()


# Flask App Globals
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


def _object_detection(image_np, min_score, draw_box):

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
            image = prepare_image(image)
            
            detection_with_filter, frame = _object_detection(image, MIN_SCORE_THRESH, DRAW_BOX)
            
            if detection_with_filter['success']:
                pass
            else:
                detection_with_filter['success'] = False
                
            resp = Response(response=json.dumps(detection_with_filter),
                                status=200,
                                mimetype="application/json")
            return resp


if __name__ == '__main__':

    detect.setup()
    time.sleep(30)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

