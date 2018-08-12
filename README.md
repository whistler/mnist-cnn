# MNIST Classifier Service
Convolutional Neural Network to classify MNIST images using Keras. It includes
a training pipeline and an inference server.

## Inference

### Run Server
Run inference server using docker. This starts the inference server on port 
5000:

    docker-compose up mnist

### Call API
Do a POST call to the server root uploading an MNIST test image to infer its 
class. The following calls the API with the test image located at 
`test/test_image.png`:

    curl -F 'image=@test/test_image.png' http://localhost:5000

Images need to be grayscale and 28px x 28px in size.

## Training
To train a new model the training pipeline can be used:

    docker-compose run mnist /code/train.py --epochs 10 --batch-size 64

## Prototype
There is a prototype notebook located [here](MNIST-CNN.ipynb)