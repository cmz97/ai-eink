# Class to monitor a rotary encoder and update a value.  You can either read the value when you need it, by calling getValue(), or
# you can configure a callback which will be called whenever the value changes.
import os
import sys
import RPi.GPIO as GPIO
import time 
import threading

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
        GPIO.add_event_detect(self.leftPin, GPIO.BOTH, callback=self.transitionOccurred, bouncetime=200)  
        # GPIO.add_event_detect(self.rightPin, GPIO.RISING, callback=self.transitionOccurred,  bouncetime=2)  

    def transitionOccurred(self, channel):
        # time.sleep(0.002) # extra 2 mSec de-bounce time
        p1 = GPIO.input(self.leftPin)
        p2 = GPIO.input(self.rightPin)
        newState = "{}{}".format(p1, p2)
        print(f"p1 {p1} p2 {p2}")
        # reject bounce
        if self.state == "01" or self.state == "10" and newState == "00":
            self.state = newState
            return

        if newState == "00" or newState ==  "11": # down
            self.direction = "D"
            self.counter += 1
        else:
            self.direction = "U"
            self.counter -=1
        self.state = newState
        if self.callback : self.callback(self.counter, self.direction)
        return


    def getValue(self):
        return self.value



class Button:

    def __init__(self, Pin, direction=None, callback=None):
        self.Pin = Pin
        self.direction = direction
        self.callback = callback
        self.last_state = None
        self.last_call = time.time()
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.Pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        # GPIO.add_event_detect(self.Pin, GPIO.BOTH, callback=self.transitionOccurred)  
        self.thread = threading.Thread(target=self.monitor_pin)
        self.thread.daemon = True  # Ensure thread exits when main program does
        self.running = True
        self.thread.start()

    def monitor_pin(self):
        while self.running:
            current_state = GPIO.input(self.Pin)
            if current_state != self.last_state:
                self.transitionOccurred()
                self.last_state = current_state
            time.sleep(0.01)  # Adjust for sensitivity vs CPU usage

    def transitionOccurred(self):
        # time.sleep(0.002) # extra 2 mSec de-bounce time
        p = GPIO.input(self.Pin)
        self.state = p
        # print(p)
        if p == 1:
            ellapse_t = time.time() - self.last_call
            self.last_call = time.time()
            print("ellapse_t", ellapse_t)
            if ellapse_t < 0.1: # reject noise
                return
            if ellapse_t < 0.5: # double click
                # self.callback(1)
                # self.shut_down()
                return 

        if self.callback and p == 1:
            print(f'{self.direction} pressed') 
            self.callback(self.direction)
        return

    def getValue(self):
        return self.value

    def shut_down(self):
        print('terminating ... ')
        # raise SystemExit
        os.system("pkill -f " + sys.argv[0])
