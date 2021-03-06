# python 2.7
import io
import socket
import struct
import numpy as np
import cv2
from time import sleep, time
from PIL import Image

from config import configs
from fcn.predict import Predictor
from util.detect import Detector
from util import img_process

IMG_W = configs['data']['image_width']
IMG_H = configs['data']['image_height']
NUM_OF_POINTS = configs['fitting']['num_of_points']
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
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image

    def listen(self):
        print('server: listening ...')
        # try:
        image_id = 0
        while True:
            new_img = self.recv_images()
            if (new_img is None):
                break
            new_img = img_process.standard_preprocess(new_img, f=False, binary=False)
            packaged_parameters = self.predict_and_fit(image_id, new_img)
            packaged_parameters_with_id = np.concatenate(([image_id], packaged_parameters))
            s_packaged_parameters = packaged_parameters_with_id.tobytes()
            self.s.sendall(s_packaged_parameters)
            image_id = image_id + 1

    def predict_and_fit(self, imageId, image):
        """ Make prediction and then fit the predicted image.
            @return: wrapped_parameters
        """
        # predict
        _start_time = time()
        predicted_image = self.predictor.predict(image)
        processed_image = img_process.standard_preprocess(predicted_image, crop=False, down=False)
        print('prediction time: ', time() - _start_time)

        # fit
        _start_time = time()
        wrapped_parameters = self.detector.get_wrapped_all_parameters(processed_image)
        print('fitting time: ', time() - _start_time)

        if configs['debug']:
            _, pt = Detector.get_distance_2_tan(wrapped_parameters[6:9])
            img1 = img_process.compute_debug_image(image, IMG_W, IMG_H, NUM_OF_POINTS, pt, wrapped_parameters)
            img2 = img_process.compute_debug_image(predicted_image, IMG_W, IMG_H, NUM_OF_POINTS, pt, wrapped_parameters)
            debug_image = np.concatenate((img1, img2), axis=0)
            img_process.show_img(debug_image)
        return wrapped_parameters

    def close_connection(self):
        self.connection.close()
        self.server.close()


if __name__ == '__main__':
    s = Server('1538680331.7627041')
    s.listen()

