#!/usr/bin/env python 
"""
Training pipeline for MNIST dataset
"""

import click
import keras
from keras.datasets import mnist
from constants import IMAGE_SIZE, NUM_CLASSES, MODEL_PATH
from model import create_model


DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 16
DEFAULT_VALIDATION_SPLIT = 0.1


def load_dataset():
    """
    Loads mnist dataset

    Returns:
        (ndarray, ndarray): (x, y) tuple with training data
        (ndarray, ndarray): (x, y) tuple with test data
    """
    return mnist.load_data()


def normalize(x):
    """
    Normalizes a array between 0 and 1

    Args:
        x (ndarray): Array to normalize

    Returns:
        ndarray: normalized array
    """
    return x.astype('float32') / 255.0


def preprocess(x_train, y_train, x_test, y_test):
    """
    Preprocess training and test data to feed into model
    """
    # Normalize to between 0 and 1
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    print('x_train max', x_train.max())
    print('x_train min', x_train.min())

    print('y_train min', y_train.min())
    print('y_train max', y_train.max())

    # one-hot encode labels
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Reshape to model input
    x_train = x_train.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    x_test = x_test.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    print('x_train shape', x_train.shape)
    print('x_test shape', x_train.shape)
    
    return x_train, y_train, x_test, y_test


def train(model, x_train, y_train, epochs=10, batch_size=16, validation_split=0.1):
    """
    Train model
    """
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        validation_split=validation_split
    )

    model.save(MODEL_PATH)


@click.command('train')
@click.option('--epochs', default=DEFAULT_EPOCHS, help='Number of epochs to train')
@click.option('--batch-size', default=DEFAULT_BATCH_SIZE, 
              help='Batch size for training')
@click.option('--validation-split', default=DEFAULT_VALIDATION_SPLIT, 
              help='Percentage of data to use for validation')
def run(epochs, batch_size, validation_split):
    """ 
    Run training pipeline
    """
    (x_train, y_train), (x_test, y_test) = load_dataset()
    x_train, y_train, x_test, y_test = preprocess(x_train, y_train, x_test, y_test)
    model = create_model()
    train(model, x_train, y_train, epochs, batch_size, validation_split)


if __name__ == '__main__':
    run()