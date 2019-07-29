# import the necessary packages
import requests
import cv2
import time

IM_WIDTH = 640
IM_HEIGHT = 480

SRC = '/home/stanlee321/Videos/camera/parking.mp4'


# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://192.168.1.2:5000/predict"
KERAS_REST_API_URL_VIDEO = "http://192.168.1.2:5000/predict_video"

IMAGE_PATH = "car.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

content_type = 'image/jpeg'
headers = {'content-type': content_type}

camera = cv2.VideoCapture(SRC)

while True:
    _, frame = camera.read()

    frame = cv2.resize(frame, (320, 240))
    
    #image = Image.fromarray(frame.astype('uint8'), 'RGB')
    _, img_encoded = cv2.imencode('.jpeg', frame)
    img_encoded.tostring()
    jpg_as_text = img_encoded.tostring()
    # load the input image and construct the payload for the request
    t1 = time.time()
    # submit the request
    r = requests.post(KERAS_REST_API_URL_VIDEO, data=jpg_as_text, headers=headers)

    t2 = time.time()

    print( t2 - t1)
    #print(r.json()) 
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
  
camera.release()
cv2.destroyAllWindows()
