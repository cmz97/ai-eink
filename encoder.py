# Class to monitor a rotary encoder and update a value.  You can either read the value when you need it, by calling getValue(), or
# you can configure a callback which will be called whenever the value changes.
import os
import sys
import RPi.GPIO as GPIO
import time 
import threading
import concurrent.futures

class Encoder:

    def __init__(self, leftPin, rightPin, callback=None):
        self.leftPin = leftPin
        self.rightPin = rightPin
        self.counter = 0
        self.state = '00'
        self.direction = None
        self.callback = callback
        self.GPIO = GPIO
        self.GPIO.setmode(GPIO.BCM)
        self.GPIO.setup(self.leftPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.GPIO.setup(self.rightPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.GPIO.add_event_detect(self.leftPin, GPIO.BOTH, callback=self.transitionOccurred, bouncetime=200)  
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
        time.sleep(1)
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
            time.sleep(0.1)  # Adjust for sensitivity vs CPU usage

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



class MultiButtonMonitor:
    def __init__(self, buttons):
        """
        buttons: A list of dictionaries, where each dictionary contains:
                 'pin': The GPIO pin number,
                 'direction': A descriptive string for the button,
                 'callback': A function to call when the button is pressed.
        """
        self.buttons = buttons
        self.GPIO = GPIO
        for button in self.buttons:
            button['last_state'] = None
            button['last_call'] = time.time()
            self.GPIO.setmode(GPIO.BCM)
            self.GPIO.setup(button['pin'], GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            time.sleep(0.1)  # Adjust for sensitivity vs CPU usage
        
        self.running = False
        self.monitor_thread = None
        self.start_monitoring()
    
    def start_monitoring(self):
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.running = True
            self.monitor_thread = threading.Thread(target=self.monitor_pins)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

    def monitor_pins(self):
        try:
            while self.running:
                for button in self.buttons:
                    time.sleep(0.1)  # Adjust for sensitivity vs CPU usage
                    self.monitor_pin(button)
                    
                time.sleep(0.1)  # Adjust for sensitivity vs CPU usage
        except Exception as e:
            print(f"An exception occurred in monitor_pins thread: {e}")
            # Handle the exception, log it, or clean up resources here

    def monitor_pin(self, button):
        current_state = self.GPIO.input(button['pin'])
        if current_state != button['last_state']:
            self.transitionOccurred(button, current_state)
            button['last_state'] = current_state

    def transitionOccurred(self, button, p):
        if p == 1:
            ellapse_t = time.time() - button['last_call']
            button['last_call'] = time.time()
            print("ellapse_t", ellapse_t)
            if ellapse_t < 0.1: # reject noise
                return
            if ellapse_t < 0.5: # double click
                # self.callback(1)
                # self.shut_down()
                return 

        if button['callback'] and p == 1:
            print(f"{button['direction']} pressed")
            button['callback'](button['direction'])
        return



    def stop_monitoring(self):
        self.running = False
        if self.monitor_thread is not None and threading.current_thread() != self.monitor_thread:
            self.monitor_thread.join()

    def shut_down(self):
        print('Terminating ...')
        self.stop_monitoring()
        os.system("pkill -f " + sys.argv[0])