from keras.models import Model
from keras.layers import concatenate, Conv2D, Dense, Flatten, Input, MaxPooling2D

from constants import IMAGE_SIZE

FILTER_SHAPE = (3, 3)


def create_model():
    """
    Creates a Convolutional Neural Network to classify MNIST images
    """
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

    conv1 = Conv2D(32, FILTER_SHAPE, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D()(conv1)
    # Based on the shape of Conv1 layers in the diagram, assuming the a Conv 
    # block in the diagram represents convolution followed by Max Pooling

    conv2_1 = Conv2D(64, FILTER_SHAPE, activation='relu', padding='same')(pool1)
    pool2_1 = MaxPooling2D()(conv2_1)
    conv2_2 = Conv2D(64, FILTER_SHAPE, activation='relu', padding='same')(pool1)
    pool2_2 = MaxPooling2D()(conv2_2)

    conv3_1 = Conv2D(256, FILTER_SHAPE, activation='relu', padding='same')(pool2_1)
    conv3_2 = Conv2D(256, FILTER_SHAPE, activation='relu', padding='same')(pool2_2)
    conv3 = concatenate([conv3_1, conv3_2])
    pool3 = MaxPooling2D()(conv3)

    flattened_conv3 = Flatten()(pool3)
    fc1 = Dense(1000, activation='relu')(flattened_conv3)
    fc2 = Dense(500, activation='relu')(fc1)

    # Using Softmax on the output layer instead of ReLU as it performs better
    # for disjoint classes
    outputs = Dense(10, activation='softmax')(fc2)

    model = Model(inputs, outputs)
    model.summary()
    return model