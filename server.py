# python 2.7
import io
import socket
import struct
import pickle
from PIL import Image

import recipe


class Server(object):
    def __init__(self):
        self.predictor = recipe.Predictior('1538586986.23')
        self.server = socket.socket()
        self.server.bind(('0.0.0.0', 8888))
        self.server.listen(0)
        print('waitting for connection')
        self.s = self.server.accept()[0]
        self.connection = self.s.makefile('rb')

    def handle(self):
        try:
            image_id = 0
            while True:
                # Read the length of the image as a 32-bit unsigned int. If the length is zero, quit the loop
                image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
                if not image_len:
                    break
                # Construct a stream to hold the image data and read the image data from the connection
                image_stream = io.BytesIO()
                image_stream.write(self.connection.read(image_len))
                # Rewind the stream, open it as an image with PIL and do some processing on it
                image_stream.seek(0)
                # image = Image.open(image_stream)
                image = Image.open(image_stream).convert('RGB')
                open_cv_image = np.array(image)

                # image.save('./comm/image-' + str(image_id) + '.png')
                # print('transmited image' + str(image_id))
                result = self.predict_and_fit(open_cv_image)
                result_with_id = [image_id, result]
                image_id = image_id + 1

                s_result = pickle.dumps(result_with_id)
                self.s.sendall(s_result)
                # self.connection.write('transmited image' + str(image_id))
        finally:
            self.connection.close()
            self.server.close()

    def predict_and_fit(self, image):
        predicted_img = self.predictor.predict('1538586986.23', image)
        pts_left, pts_right = recipe.fit(predicted_img)
        return np.array([pts_left, pts_right])


if __name__ == '__main__':
    s = Server()
    s.handle()

