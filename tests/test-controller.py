from time import sleep

from .context import control
from control.controller import Controller


def test_controller():
    cont = Controller()
    cont.start()
    sleep(1)
    # turn right!
    print('turning right')
    cont.make_decision(12.0, 12.0, 0)
    sleep(10)
    cont.finish_control()
    cont.cleanup()


if __name__ == '__main__':
    test_controller()
