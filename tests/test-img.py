from util import img_process
import cv2

def test_binarize():
    for i in range(1, 100):
        img = cv2.imread('./tests/input/' + str(i) + '.png')
        # img = img_process.binarize(img)
        img = img_process.birdeye(img)
        img = img_process.standard_preprocess(img, False, False, True, True)
        cv2.imshow('test', img)
        cv2.waitKey(0)


def main():
    test_binarize()


if __name__ == '__main__':
    main()
