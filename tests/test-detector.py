from os.path import join
from .context import util
from util.detect import Detector
from util import img_process

BASE_DIR = join('.', 'tests')

def test_detector():
    det = Detector()
    test_img = img_process.img_load(join(BASE_DIR, 'input', '1.png'))
    test_img = img_process.standard_preprocess(test_img, crop=False, down=False, f=True, binary=True)
    paras = det.get_wrapped_all_parameters(test_img)
    dis, index = Detector.get_distance_2_tan(paras[3:6])
    print(dis, index)


if __name__ == '__main__':
    test_detector()
