# some classical recipes
import sys
from os.path import join
from .context import util
from .context import car
from .context import config
from util.detect import Detector
from util import img_process
from car import Car
from config import configs

def test_offline():
    try:
        car = Car()
        car.run_offline()
    except KeyboardInterrupt:
        car.stop()


def test_online_multithread(ip, addr):
    try:
        car = Car()
        car.run_online(ip, addr)
    except KeyboardInterrupt:
        car.stop()


def test_online_single(ip, addr):
    try:
        car = Car()
        car.run_online_single(ip, addr)
    except KeyboardInterrupt:
        car.stop()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please provide running mode')
    else:
        mode = str(sys.argv[1])
        print('running mode: ' + str(mode))
        if mode == 'online':
            test_online_multithread(configs['server']['ip'], configs['server']['port'])
        elif mode == 'online-single':
            test_online_single(configs['server']['ip'], configs['server']['port'])
        elif mode == 'offline':
            test_offline()
        else:
            print('mode can only be online or offline')

