import RPi.GPIO as GPIO
from time import sleep


class Motor(object):
    def __init__(self, slient=False):
        """ Set up GPIO environment.
        """
        self.initGPIO()
        self._slient = slient

    def initGPIO(self):
        GPIO.setmode(GPIO.BCM)
        ENA, ENB = 26, 11
        IN1, IN2, IN3, IN4 = 19, 13, 6, 5
        sleep(1)
        #  Motor Pins
        GPIO.setup(ENA, GPIO.OUT) # ENA
        GPIO.setup(ENB, GPIO.OUT) # ENB
        GPIO.setup(IN1, GPIO.OUT) # IN1
        GPIO.setup(IN2, GPIO.OUT) # IN2
        GPIO.setup(IN3, GPIO.OUT) # IN3
        GPIO.setup(IN4, GPIO.OUT) # IN4

        # PWM pin and Frequency
        self.pwmR = GPIO.PWM(26, 100)
        self.pwmL = GPIO.PWM(11, 100)
        self.pwmR.start(0)
        self.pwmL.start(0)
        sleep(1)

        GPIO.output(19, GPIO.HIGH)
        GPIO.output(13, GPIO.LOW)
        GPIO.output(6, GPIO.HIGH)
        GPIO.output(5, GPIO.LOW)
        sleep(1)
        print ('GPIO INITIALIZED')

    def motor_startup(self):
        self.pwmL.ChangeDutyCycle(30)
        self.pwmR.ChangeDutyCycle(30)

    def motor_stop(self):
        self.pwmL.stop()
        self.pwmR.stop()

    def motor_set_new_speed(self, left, right):
        self.pwmL.ChangeDutyCycle(left)
        self.pwmR.ChangeDutyCycle(right)
        if not self._slient:
            print('Motor: ', 'pwm_l_new', left, 'pwm_r_new', right)

    def motor_cleanup(self):
        GPIO.cleanup()
