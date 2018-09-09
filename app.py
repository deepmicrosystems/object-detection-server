import numpy as np
from flask import Flask, render_template, request, jsonify, Response
import io
from PIL import Image
import json
import time
from libs.image_processor.imageprocessor import ImageProcessor

index_to_string = {
    3: 'car',
    6: 'bus',
    8: 'truck',
    1: 'person',
    10: "traffic light"
}

# Some globals
detect = ImageProcessor(path_to_model='ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb',
                        path_to_labels='libs/object_detection/data/mscoco_label_map.pbtxt',
                        model_name='ssdlite_mobilenet_v2_coco_2018_05_09')
# Flask App Globals
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


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

            # classify the input image and then initialize the list
            # of predictions to return to the client
            (boxes, scores, classes, num) = detect.detect(image)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            # Filter just car detections.
            for i, b in enumerate(boxes[0]):
                if int(classes[0][i]) in index_to_string:
                    if scores[0][i] >= 0.3:

                        x0 = int(boxes[0][i][3] * image.shape[1])
                        y0 = int(boxes[0][i][2] * image.shape[0])

                        x1 = int(boxes[0][i][1] * image.shape[1])
                        y1 = int(boxes[0][i][0] * image.shape[0])

                        r = {
                            'coord': {
                                'xmin': x0, 'ymin': y0,
                                'xmax': x1, 'ymax': y1
                            },
                            'class': index_to_string[int(classes[0][i])],
                            'probability': float(scores[0][i])
                        }

                        data["predictions"].append(r)

                # indicate that the request was a success
                data["success"] = True
                resp = Response(response=json.dumps(data),
                                status=200,
                                mimetype="application/json")
                return resp


if __name__ == '__main__':

    detect.setup()
    time.sleep(30)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

