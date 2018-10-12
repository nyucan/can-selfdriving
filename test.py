from util import util
from os.path import join
import cv2

from fcn.predict import Predictor
from util.detect import Detector


def test_predictor(img):
    test_predictor = Predictor('1538586986.23')
    predicted_img = test_predictor.predict(img)
    cv2.imwrite(join('.', 'test-output', 'test-predict.png'), predicted_img)
    return predicted_img


def test_detector(img):
    test_detector = Detector()
    paras = test_detector.get_wrapped_all_parameters(img)
    print(paras)
    detectored_img = test_detector.mark_image_with_parameters(img, paras)
    # test_detector.visualization(detectored_img)
    cv2.imwrite(join('.', 'test-output', 'test-predict-2.png'), detectored_img)
    return detectored_img


def main():
    test_img = util.get_an_image_from(join('.', 'data', 'testing', 'image', '0.png'))
    p_img = test_predictor(test_img)
    d_img = test_detector(p_img)
    # Detector.visualization(d_img)


if __name__ == '__main__':
    main()
