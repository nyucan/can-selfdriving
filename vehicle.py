# python 2.7
# run on the raspberry pi cars
from __future__ import absolute_import, division, print_function
from os.path import join

from util import util
from util import detect
from recipe import test_model

testing_dir = join('.', 'data', 'testing')


def fit_lane():
    print('fitting lane')
    detect.mark_images_from(join(testing_dir, 'predict'), join(testing_dir, 'fitting'))


def main():
    ## TODO take picture with camera
    ## TODO crop image
    ## generate the predicted image with trained model
    # test_model('1538068266.3')
    test_model('1538680331.7627041') # model trained with color shiftting
    ## fit the lane with curve
    fit_lane()
    ## TODO decide how to move


if __name__ == '__main__':
    main()


