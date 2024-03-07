# Class to monitor a rotary encoder and update a value.  You can either read the value when you need it, by calling getValue(), or
# you can configure a callback which will be called whenever the value changes.

import RPi.GPIO as GPIO
import time 

class Encoder:

    def __init__(self, leftPin, rightPin, callback=None):
        self.leftPin = leftPin
        self.rightPin = rightPin
        self.counter = 0
        self.state = '00'
        self.direction = None
        self.callback = callback
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.leftPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.rightPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.leftPin, GPIO.RISING, callback=self.transitionOccurred, bouncetime=200)  
        # GPIO.add_event_detect(self.rightPin, GPIO.RISING, callback=self.transitionOccurred,  bouncetime=2)  

    def transitionOccurred(self, channel):
        time.sleep(0.002) # extra 2 mSec de-bounce time
        p1 = GPIO.input(self.leftPin)
        p2 = GPIO.input(self.rightPin)
        newState = "{}{}".format(p1, p2)
        # print(f"p1 {p1} p2 {p2}")
        if newState == "00" or newState ==  "11": # down
            self.direction = "D"
            self.counter += 1
        else:
            self.direction = "U"
            self.counter -=1
        if self.callback : self.callback(self.counter, self.direction)
        return


    def getValue(self):
        return self.value



class Button:

    def __init__(self, Pin, callback=None):
        self.Pin = Pin
        self.value = 0
        self.state = '0'
        self.direction = None
        self.callback = callback
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.Pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.add_event_detect(self.Pin, GPIO.BOTH, callback=self.transitionOccurred)  

    def transitionOccurred(self, channel):
        # time.sleep(0.002) # extra 2 mSec de-bounce time
        p = GPIO.input(self.Pin)
        self.state = p
        # print(p)
        if self.callback and p == 1: self.callback()
        return

    def getValue(self):
        return self.value
