# python 2.7
import io
import socket
import struct
import numpy as np
import cv2
from time import sleep
from PIL import Image

from fcn.predict import Predictor
from util.detect import Detector
from util import img_process

LOW_LANE_COLOR = np.uint8([[[0,0,0]]])
UPPER_LANE_COLOR = np.uint8([[[0,0,0]]]) + 10

class Server(object):
    def __init__(self, model):
        self.predictor = Predictor(model)
        self.detector = Detector()
        self.server = socket.socket()
        self.server.bind(('0.0.0.0', 8888))
        self.server.listen(0)
        print('server: waitting for connection')

        self.s = self.server.accept()[0]
        self.connection = self.s.makefile('rb')
        print('server: new connection')

    @staticmethod
    def preprocess_image(image):
        """ Perform filter operations to pre-process the image.
        """
        img_cropped = img_process.crop_image(image, 0.45, 0.85)
        # print(img_cropped.shape)
        # img_downsampled = img_process.down_sample(img_cropped, (160, 48))
        # lane_img = img_process.lane_filter(img_downsampled, LOW_LANE_COLOR, UPPER_LANE_COLOR)
        # bin_img = lane_img / 255
        return img_cropped

    def recv_images(self):
        """ Get image from the server.
            @returns
                return `None` if no more images
        """
        image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            print('server: receive None')
            return None
        # Construct a stream to hold the image data and read the image data from the connection
        image_stream = io.BytesIO()
        image_stream.write(self.connection.read(image_len))
        image_stream.seek(0)
        image = Image.open(image_stream).convert('RGB')
        open_cv_image = np.array(image)
        processed_image = Server.preprocess_image(open_cv_image)
        return processed_image

    def listen(self):
        print('server: listening ...')
        # try:
        image_id = 0
        while True:
            new_img = self.recv_images()
            if (new_img is None):
                break
            print('Server: transmited image ' + str(image_id))
            packaged_parameters = self.predict_and_fit(image_id, new_img)
            print('Server: predicted and fitted image ' + str(image_id))
            packaged_parameters_with_id = np.concatenate(([image_id], packaged_parameters))
            s_packaged_parameters = packaged_parameters_with_id.tobytes()
            self.s.sendall(s_packaged_parameters)
            image_id = image_id + 1
        # except:
        #     print('closed by thread')
        # finally:
        #     self.close_connection()
        #     print('connection closed')

    def predict_and_fit(self, imageId, image):
        """ Make prediction and then fit the predicted image.
            @return: image, left_parameters, left_parameters
        """
        img_process.img_save(image, './test-output/online/1/' + str(imageId) + '.png')
        # predict
        predicted_img = self.predictor.predict(image)
        img_process.img_save(predicted_img, './test-output/online/2/' + str(imageId) + '.png')

        predicted_img = img_process.lane_filter(predicted_img, LOW_LANE_COLOR, UPPER_LANE_COLOR)
        predicted_img = predicted_img / 255

        # fit
        wrapped_parameters = self.detector.get_wrapped_all_parameters(predicted_img)

        debug_img = Detector.mark_image_with_parameters(image, wrapped_parameters)
        img_process.img_save(debug_img, './test-output/online/3/' + str(imageId) + '.png')
        return wrapped_parameters

    def close_connection(self):
        self.connection.close()
        self.server.close()

    @classmethod
    def crop_image(cls, img, lower_bound, upper_bound):
        """ Crop an image in the vertical direction with lower and upper bound.
        """
        img_cropped = img[int(img.shape[1]*lower_bound):int(img.shape[1]*upper_bound),:]
        return img_cropped


if __name__ == '__main__':
    s = Server('1538680331.7627041')
    s.listen()

