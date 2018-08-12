import numpy as np
import train
import model


def test_load_data():
    (x_train, y_train), (x_test, y_test) = train.load_dataset()
    assert x_train.shape == (60000, 28, 28)
    assert y_train.shape == (60000,)
    assert x_test.shape == (10000, 28, 28)
    assert y_test.shape == (10000,)
    

def test_normalize():
    x = np.random.rand(1000, 10, 10) * 255
    x_int = x.astype(np.int)
    x_norm = train.normalize(x_int)
    epsilon = 0.01
    assert x_norm.min() < epsilon
    assert x_norm.max() > 1 - epsilon

def test_preprocess():
    x = np.array([np.random.rand(28, 28)])
    y = np.array([2])
    x_train_pp, y_train_pp, x_test_pp, y_test_pp = train.preprocess(x, y, x, y)
    assert y_train_pp.shape == (1, 10)
    assert y_test_pp.shape == (1, 10)
    assert x_train_pp.shape == (1, 28, 28, 1)
    assert x_test_pp.shape == (1, 28, 28, 1)

def test_train():
    cnn = model.create_model()
    x_train = np.random.rand(1, 28, 28, 1)
    y_train = np.array([[0]*10])
    y_train[0] = 1
    train.train(cnn, x_train, y_train, epochs=1, validation_split=0.0)
