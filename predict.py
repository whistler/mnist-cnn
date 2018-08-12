"""
HTTP Service for inference of MNIST digits
"""

import numpy as np
import os
import imageio
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from constants import IMAGE_SIZE, MODEL_PATH
from api_error import APIError


app = Flask(__name__)
basedir = os.path.dirname(__file__)
model = load_model(os.path.join(basedir, MODEL_PATH))

# This is needed for multi-threading to work. Issue described here: 
# https://github.com/keras-team/keras/issues/2397
graph = tf.get_default_graph()

ERROR_SINGLE_IMAGE_ONLY = 'Prediction accepts a single image file upload'
ERROR_INVALID_IMAGE_FORMAT = 'Unable to read image'
ERROR_INVALID_SIZE_OR_CHANNELS = 'Image needs to be grayscale with size of 28 x 28 pixels.'


@app.route('/', methods=['POST'])
def predict():
    # global is needed as a workaround for multi-threading issue. See above.
    global graph

    if len(request.files) != 1:
        raise APIError(ERROR_SINGLE_IMAGE_ONLY)

    files = request.files.to_dict()
    file_object = list(files.values())[0]
    _, image_format = os.path.splitext(file_object.filename)
    
    try:
        image = imageio.imread(file_object, image_format)
    except Exception as e:
        print(file_object.content_length)
        raise APIError(ERROR_INVALID_IMAGE_FORMAT)

    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        # Assuming test images from MNIST dataset will be used. We can add 
        # ability to resize images if images of a different size are used.
        raise APIError(ERROR_INVALID_SIZE_OR_CHANNELS)

    image_normalized = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1)) / 255
    
    with graph.as_default():
        predicted_classes = model.predict(image_normalized)

    digit = np.argmax(predicted_classes[0])
    probability = predicted_classes.max()
    return jsonify(digit=int(digit), probability=float(probability))


@app.errorhandler(APIError)
def handle_invalid_usage(error):
    """
    Convert APIError exceptions to JSON responses
    """
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response