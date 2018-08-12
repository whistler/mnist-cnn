import io
import numpy as np
import os
import predict
import imageio


test_path = os.path.dirname(__file__)

def test_predict_test_image():
    test_image_path = os.path.join(test_path, 'test_image.png')
    client = predict.app.test_client()

    with open(test_image_path, 'rb') as fp:
        test_image = fp.read()
        test_image_io = io.BytesIO(test_image)

        response = client.post('/', data={
            'image': (test_image_io, 'test_image.png'),
        })

    assert response.status_code == 200
    result = response.get_json()
    assert 'digit' in result
    assert 'probability' in result
    assert result['digit'] == 7
    assert result['probability'] > 0.5


def test_predict_no_image():
    client = predict.app.test_client()
    response = client.post('/', data={'bad_input': 0})
    assert response.status_code == 400
    assert response.json.get('message') == predict.ERROR_SINGLE_IMAGE_ONLY


def test_predict_invalid_format_image():
    client = predict.app.test_client()
    invalid_image = io.BytesIO(b'Not a real image')
    response = client.post('/', data={
        'image': (invalid_image, 'test_image.png'),
    })
    assert response.status_code == 400
    assert response.json.get('message') == predict.ERROR_INVALID_IMAGE_FORMAT


def test_predict_invalid_shape_image():
    client = predict.app.test_client()
    small_image = np.random.rand(10,10)
    image = io.BytesIO()
    imageio.imsave(image, small_image, 'png')
    image.seek(0)
    response = client.post('/', data={
        'image': (image, 'test_image.png'),
    })
    assert response.status_code == 400
    assert response.json.get('message') == predict.ERROR_INVALID_SIZE_OR_CHANNELS
