# FlaskObjectDetection - TensorFlow

[![](images/logo.png)](https://www.tensorflow.org/)

## USE

### Serser side.

Run the script:

```
$ python app.py
```

for start the server, the server will be available in `http://localhost:5000/predict` waiting for the POST requests.


### Client side.

Use the `client.py` 

```python
# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "car.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was successful
if r["success"]:
    # loop over the predictions and display them
    print(r)

# otherwise, the request failed
else:
    print("Request failed")
```

...or from the terminal with curl:

```console
$ curl -X POST -F image=@car.jpg 'http://localhost:5000/predict'
```
