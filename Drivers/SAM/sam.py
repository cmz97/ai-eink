import serial
import time

# Setup the serial connection
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

def process_button_state(state):
    """Process the button state byte."""
    if state & EncodeTable["BTN_UP"]:
        print("Button UP pressed")
    if state & EncodeTable["BTN_DOWN"]:
        print("Button DOWN pressed")
    if state & EncodeTable["BTN_SELECT"]:
        print("Button SELECT pressed")

# Dictionary for button encoding
EncodeTable = {"BTN_UP": 0b001, "BTN_DOWN": 0b010, "BTN_SELECT": 0b100}

try:
    while True:
        line = ser.readline().decode().strip()  # Read a line from the serial port
        if line.isdigit():  # Check if the line is a number
            button_state = int(line)
            process_button_state(button_state)
        else:
            print(line)  # Print any other messages such as "xPOWER_ON", "xPOWER_OFF", "xUP"

except KeyboardInterrupt:
    print("Program terminated by user")

finally:
    ser.close()  # Close the serial port when done

