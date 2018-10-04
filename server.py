# python 2.7
import io
import socket
import struct
from PIL import Image


class Server(object):
    def __init__(self):
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
                # Read the length of the image as a 32-bit unsigned int. If the
                # length is zero, quit the loop
                image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
                if not image_len:
                    break
                # Construct a stream to hold the image data and read the image
                # data from the connection
                image_stream = io.BytesIO()
                image_stream.write(self.connection.read(image_len))
                # Rewind the stream, open it as an image with PIL and do some
                # processing on it
                image_stream.seek(0)
                image = Image.open(image_stream)

                # TODO op the image
                image.save('./comm/image-' + str(image_id) + '.png')
                image_id = image_id + 1
                print('transmited image' + str(image_id))

                # TODO return the result
                self.s.sendall('transmited image' + str(image_id))
                # self.connection.write('transmited image' + str(image_id))
        finally:
            self.connection.close()
            self.server.close()


if __name__ == '__main__':
    s = Server()
    s.handle()

