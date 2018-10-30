from os.path import join
from .context import util
from .context import config
from util.detect import Detector
from util import img_process
from config import configs

BASE_DIR = join('.', 'tests')
IMG_W = configs['data']['image_width']
IMG_H = configs['data']['image_height']
NUM_OF_POINTS = configs['fitting']['num_of_points']

def test_imshow():
    for i in range(1, 140):
        test_img = img_process.img_load(join(BASE_DIR, 'input', str(i) + '.png'))
        img_process.show_img(test_img)


def test_detector():
    det = Detector()
    for i in range(1, 140):
        ori_img = img_process.img_load(join(BASE_DIR, 'input', str(i) + '.png'))
        test_img = img_process.standard_preprocess(ori_img, crop=False, down=False, f=True, binary=True)
        paras = det.get_wrapped_all_parameters(test_img)
        dis, pt = Detector.get_distance_2_tan(paras[6:9])
        fitting_img = img_process.mark_image_with_parameters(ori_img, paras, IMG_H, NUM_OF_POINTS)
        img_process.mark_image_with_pt(fitting_img, (80, 24), (0,255,0))
        img_process.mark_image_with_pt(fitting_img, pt, (0, 255, 255))
        img_process.img_save(fitting_img, join(BASE_DIR, 'output', str(i) + '.png'))
        print(dis, pt)


if __name__ == '__main__':
    test_detector()
