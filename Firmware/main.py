import machine
import utime

#Instruction Set
EncodeTable = {"BTN_UP": 0b1, "BTN_DOWN": 0b10, "BTN_SELECT": 0b100}

# Set up GPIO pins
selectBTN = machine.Pin(16, machine.Pin.IN, machine.Pin.PULL_DOWN)
upBTN = machine.Pin(17, machine.Pin.IN, machine.Pin.PULL_DOWN)
downBTN = machine.Pin(18, machine.Pin.IN, machine.Pin.PULL_DOWN)
powerStatus = machine.Pin(21, machine.Pin.OUT)
einkStatus = machine.Pin(9, machine.Pin.OUT)

powerStatus.low()
powerStatus.low()
einkStatus.high()

# Debounce time in milliseconds
debounce_time = 50

# Function to debounce button press
def debounce(pin):
    state = pin.value()
    utime.sleep_ms(debounce_time)
    if pin.value() != state:
        return False
    return True

def get_debounced_state(pin):
    return pin.value() and debounce(pin) 

def send_button_state():
    state_byte = 0
    state_byte |= get_debounced_state(selectBTN) * EncodeTable["BTN_SELECT"]
    state_byte |= get_debounced_state(upBTN)  * EncodeTable["BTN_UP"]
    state_byte |= get_debounced_state(downBTN) * EncodeTable["BTN_DOWN"]
    print(f"{state_byte}\n")

        
# Interrupt handler for down button
def button_handler(pin):
    if debounce(pin):
        send_button_state()



# Set up interrupt handlers
selectBTN.irq(trigger=machine.Pin.IRQ_RISING | machine.Pin.IRQ_FALLING, handler=button_handler)
upBTN.irq(trigger=machine.Pin.IRQ_RISING | machine.Pin.IRQ_FALLING, handler=button_handler)
downBTN.irq(trigger=machine.Pin.IRQ_RISING | machine.Pin.IRQ_FALLING, handler=button_handler)

while True:
    if debounce(selectBTN) and selectBTN.value() == 1 and upBTN.value() == 0 and downBTN.value() == 0:
        start_time = utime.ticks_ms()
        while utime.ticks_diff(utime.ticks_ms(), start_time) < 2000:
            if selectBTN.value() == 0:
                break
            utime.sleep_ms(10)
        if utime.ticks_diff(utime.ticks_ms(), start_time) >= 2000 and powerStatus.value() == 0:
            powerStatus.high()
            einkStatus.low() # inverted logic
            print("xPOWER_ON")

    if debounce(upBTN) and selectBTN.value() == 1:
        print("xUP")
        start_time = utime.ticks_ms()
        while utime.ticks_diff(utime.ticks_ms(), start_time) < 5000:
            if upBTN.value() == 0 or selectBTN.value() == 0:
                break
            utime.sleep_ms(10)
        if utime.ticks_diff(utime.ticks_ms(), start_time) >= 5000 and powerStatus.value() == 1:
            powerStatus.low()
            einkStatus.high() # inverted logic
            print("xPOWER_OFF")
    utime.sleep_ms(1)




