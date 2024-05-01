import serial
import time
import threading


class SAM:
    def __init__(self,callback):
        
        self.EncodeTable = {"BTN_UP": 0b001, "BTN_DOWN": 0b010, "BTN_SELECT": 0b100}
        self.callback = callback
        self.running = False
        self.monitor_thread = None
        self.start_monitoring()
    
    def process_button_state(self, state):
        """Process the button state byte."""
        if state & self.EncodeTable["BTN_UP"]:
           self.callback(0)
        if state & self.EncodeTable["BTN_DOWN"]:
            self.callback(1)
        if state & self.EncodeTable["BTN_SELECT"]:
            self.callback(2)


    def start_monitoring(self):
        self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.running = True
            self.monitor_thread = threading.Thread(target=self.monitor_pins)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

    def monitor_pins(self):
        try:
            while True:
                print("SAM Ping...")
                line = self.ser.readline().decode().strip()  # Read a line from the serial port
                if line.isdigit():  # Check if the line is a number
                    button_state = int(line)
                    self.process_button_state(button_state)
                else:
                    print(f"SAM INFO MESSAGE: {line}")  # Print any other messages such as "xPOWER_ON", "xPOWER_OFF", "xUP"
                    print(f"SAM MESSAGE LENTGH: {len(line)}")
        except Exception as e:
            print(f"An exception occurred in SAM thread: {e}")
            # Handle the exception, log it, or clean up resources here


   
    def stop_monitoring(self):
        self.running = False
        if self.monitor_thread is not None and threading.current_thread() != self.monitor_thread:
            self.monitor_thread.join()
        self.ser.close()
