from util import util
from os.path import join
from time import sleep
import cv2

from fcn.predict import Predictor
from fcn.trainer import Trainer
from util.detect import Detector
from control.controller import Controller


def test_predictor(img):
    test_predictor = Predictor('1538586986.23')
    predicted_img = test_predictor.predict(img)
    return predicted_img


def test_detector(img):
    test_detector = Detector()
    paras = test_detector.get_wrapped_all_parameters(img)
    detectored_img = Detector.mark_image_with_parameters(img, paras)
    return detectored_img


def test_predictor_and_detector():
    test_img = util.get_an_image_from(join('.', 'data', 'testing', 'image', '0.png'))
    p_img = test_predictor(test_img)
    d_img = test_detector(p_img)

    cv2.imwrite(join('.', 'test-output', 'test-predict.png'), p_img)
    cv2.imwrite(join('.', 'test-output', 'test-predict-2.png'), d_img)

    Detector.visualization(d_img)
    cv2.imwrite(join('.', 'test-output', 'test-predict-3.png'), d_img)


def test_trainer():
    nn = Trainer.train_nn_from_sketch()


def test_controller():
    cont = Controller()
    cont.motor_startup()
    sleep(5)
    cont.motor_set_new_speed(60, 30) # turn right
    sleep(5)
    cont.motor_set_new_speed(30, 60) # turn left
    sleep(5)
    cont.motor_stop()


def main():
    ## 1. Trainer
    # test_trainer()
    ## 2. Predictor and Detector
    # test_predictor_and_detector()
    ## 3. Controller
    # test_controller()
    # sleep(5)


if __name__ == '__main__':
    main()
