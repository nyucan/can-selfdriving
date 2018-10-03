# python 2.7
from __future__ import absolute_import, division, print_function

import RPi.GPIO as GPIO
from time import sleep


class MotorController(object):
    def __init__(self):
        self.perpare()


    def perpare(self):
        GPIO.setmode(GPIO.BCM)
        # Setup GPIO Pins
        GPIO.setup(26, GPIO.OUT) #ENA
        GPIO.setup(11, GPIO.OUT) #ENB
        GPIO.setup(19, GPIO.OUT) #IN1
        GPIO.setup(13, GPIO.OUT) #IN2
        GPIO.setup(6, GPIO.OUT)  #IN3
        GPIO.setup(5, GPIO.OUT)  #IN4

        # Set PWM instance and their frequency
        self.pwmR = GPIO.PWM(26, 100)
        self.pwmL = GPIO.PWM(11, 100)
        sleep(1)

        self.pwmR.start(0)
        self.pwmL.start(0)
        GPIO.output(19, GPIO.LOW)
        GPIO.output(13, GPIO.HIGH)
        GPIO.output(6, GPIO.LOW)
        GPIO.output(5, GPIO.HIGH)
        sleep(1)


    def run(self, speed=100):
        self.pwmR.ChangeDutyCycle(speed)
        self.pwmL.ChangeDutyCycle(speed)


    def turn(self, degree):
        pass


    def stop(self):
        self.pwmR.stop()
	    self.pwmL.stop()
        GPIO.cleanup()
