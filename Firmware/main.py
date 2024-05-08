import machine
import utime

class einkDSP:
    def __init__(self) -> None:

        self.LUT_ALL = [
        0x01,	0x05,	0x20,	0x19,	0x0A,	0x01,	0x01,	
        0x05,	0x0A,	0x01,	0x0A,	0x01,	0x01,	0x01,	
        0x05,	0x09,	0x02,	0x03,	0x04,	0x01,	0x01,	
        0x01,	0x04,	0x04,	0x02,	0x00,	0x01,	0x01,	
        0x01,	0x00,	0x00,	0x00,	0x00,	0x01,	0x01,	
        0x01,	0x00,	0x00,	0x00,	0x00,	0x01,	0x01,	
        0x01,	0x05,	0x20,	0x19,	0x0A,	0x01,	0x01,	
        0x05,	0x4A,	0x01,	0x8A,	0x01,	0x01,	0x01,	
        0x05,	0x49,	0x02,	0x83,	0x84,	0x01,	0x01,	
        0x01,	0x84,	0x84,	0x82,	0x00,	0x01,	0x01,	
        0x01,	0x00,	0x00,	0x00,	0x00,	0x01,	0x01,	
        0x01,	0x00,	0x00,	0x00,	0x00,	0x01,	0x01,	
        0x01,	0x05,	0x20,	0x99,	0x8A,	0x01,	0x01,	
        0x05,	0x4A,	0x01,	0x8A,	0x01,	0x01,	0x01,	
        0x05,	0x49,	0x82,	0x03,	0x04,	0x01,	0x01,	
        0x01,	0x04,	0x04,	0x02,	0x00,	0x01,	0x01,	
        0x01,	0x00,	0x00,	0x00,	0x00,	0x01,	0x01,	
        0x01,	0x00,	0x00,	0x00,	0x00,	0x01,	0x01,	
        0x01,	0x85,	0x20,	0x99,	0x0A,	0x01,	0x01,	
        0x05,	0x4A,	0x01,	0x8A,	0x01,	0x01,	0x01,	
        0x05,	0x49,	0x02,	0x83,	0x04,	0x01,	0x01,	
        0x01,	0x04,	0x04,	0x02,	0x00,	0x01,	0x01,	
        0x01,	0x00,	0x00,	0x00,	0x00,	0x01,	0x01,	
        0x01,	0x00,	0x00,	0x00,	0x00,	0x01,	0x01,	
        0x01,	0x85,	0xA0,	0x99,	0x0A,	0x01,	0x01,	
        0x05,	0x4A,	0x01,	0x8A,	0x01,	0x01,	0x01,	
        0x05,	0x49,	0x02,	0x43,	0x04,	0x01,	0x01,	
        0x01,	0x04,	0x04,	0x42,	0x00,	0x01,	0x01,	
        0x01,	0x00,	0x00,	0x00,	0x00,	0x01,	0x01,	
        0x01,	0x00,	0x00,	0x00,	0x00,	0x01,	0x01,	
        0x09,	0x10,	0x3F,	0x3F,	0x00,	0x0B,		
        ]
        self.emptyImage = [0xFF] * 24960
        self.oldData = [0] * 12480
        

        #Pin Def

        self.DC_PIN = 6
        self.RST_PIN = 13
        self.BUSY_PIN = 9

        self.EPD_WIDTH = 240 
        self.EPD_HEIGHT = 416 

    
        self.GPIO = GPIO

        self.spi = self.EPD_GPIO_Init()
        self.epd_w21_init_4g()


    def EPD_GPIO_Init(self):
        # GPIO.cleanup()
        self.GPIO.setwarnings(False) 
        self.GPIO.setmode(GPIO.BCM)
        self.GPIO.setup(self.DC_PIN, GPIO.OUT) #DC 
        self.GPIO.setup(self.RST_PIN, GPIO.OUT) #REST
        self.GPIO.setup(self.BUSY_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP) #BUSY
      



        bus = 0 # We only have SPI bus 0 available to us on the Pi
        device = 0 #Device is the chip select pin. Set to 0 or 1, depending on the connections
        spi = spidev.SpiDev()
        spi.open(bus, device) 
        spi.max_speed_hz = 30000000 #30000000MHZ
        spi.mode = 0
        return spi

    def SPI_Delay(self):
        time.sleep(0.000001) #0.000001

    def SPI_Write(self,value):
        return self.spi.xfer2([value])

    def epd_w21_write_cmd(self,command):
        self.SPI_Delay()
        self.GPIO.output(self.DC_PIN, GPIO.LOW)
        self.SPI_Write(command)

    def epd_w21_write_data(self,data):
        self.SPI_Delay()
        self.GPIO.output(self.DC_PIN, GPIO.HIGH)
        self.SPI_Write(data)

    def delay_xms(self,xms):
        time.sleep(xms / 1000.0)

    def epd_w21_init(self):
        self.delay_xms(100)  # At least 10ms delay
        self.GPIO.output(self.RST_PIN, GPIO.LOW)
        self.delay_xms(20)
        self.GPIO.output(self.RST_PIN, GPIO.HIGH)
        self.delay_xms(20)

   

    def EPD_Display(self,image):
        width = (self.EPD_WIDTH + 7) // 8
        height = self.EPD_HEIGHT

        self.epd_w21_write_cmd(0x10)
        for j in range(height):
            for i in range(width):
                self.epd_w21_write_data(image[i + j * width])

        self.epd_w21_write_cmd(0x13)
        for _ in range(height * width):
            self.epd_w21_write_data(0x00) 

        self.epd_w21_write_cmd(0x12)
        self.delay_xms(1)  # Necessary delay
        # You would need to implement self.lcd_chkstatus here

    def lcd_chkstatus(self):
        while self.GPIO.input(self.BUSY_PIN) == GPIO.LOW:  # Assuming LOW means busy
            time.sleep(0.01)  # Wait 10ms before checking again

    def epd_sleep(self):
        self.epd_w21_write_cmd(0x02)  # Power off
        self.lcd_chkstatus()  # Implement this to check the display's busy status
        
        self.epd_w21_write_cmd(0x07)  # Deep sleep
        self.epd_w21_write_data(0xA5)

    def epd_init(self):
        self.epd_w21_init()  # Reset the e-paper display
        
        self.epd_w21_write_cmd(0x04)  # Power on
        self.lcd_chkstatus()  # Implement this to check the display's busy status

        self.epd_w21_write_cmd(0x50)  # VCOM and data interval setting
        self.epd_w21_write_data(0x97)  # Settings for your display

    def epd_init_fast(self):
        self.epd_w21_init()  # Reset the e-paper display
        
        self.epd_w21_write_cmd(0x04)  # Power on
        self.lcd_chkstatus()  # Implement this to check the display's busy status

        self.epd_w21_write_cmd(0xE0)
        self.epd_w21_write_data(0x02)

        self.epd_w21_write_cmd(0xE5)
        self.epd_w21_write_data(0x5A)

    def epd_init_part(self):
        self.epd_w21_init()  # Reset the e-paper display
        
        self.epd_w21_write_cmd(0x04)  # Power on
        self.lcd_chkstatus()  # Implement this to check the display's busy status

        self.epd_w21_write_cmd(0xE0)
        self.epd_w21_write_data(0x02)

        self.epd_w21_write_cmd(0xE5)
        self.epd_w21_write_data(0x6E)

        self.epd_w21_write_cmd(0x50)
        self.epd_w21_write_data(0xD7)

    def power_off(self):
        self.epd_w21_write_cmd(0x02)
        self.lcd_chkstatus()

    def write_full_lut(self):
        self.epd_w21_write_cmd(0x20)  # Write VCOM register
        for i in range(42):
            self.epd_w21_write_data(self.LUT_ALL[i])

        self.epd_w21_write_cmd(0x21)  # Write LUTWW register
        for i in range(42, 84):
            self.epd_w21_write_data(self.LUT_ALL[i])

        self.epd_w21_write_cmd(0x22)  # Write LUTR register
        for i in range(84, 126):
            self.epd_w21_write_data(self.LUT_ALL[i])

        self.epd_w21_write_cmd(0x23)  # Write LUTW register
        for i in range(126, 168):
            self.epd_w21_write_data(self.LUT_ALL[i])

        self.epd_w21_write_cmd(0x24)  # Write LUTB register
        for i in range(168, 210):
            self.epd_w21_write_data(self.LUT_ALL[i])

    def epd_w21_init_4g(self):
        self.epd_w21_init()  # Reset the e-paper display

        # Panel Setting
        self.epd_w21_write_cmd(0x00)
        self.epd_w21_write_data(0xFF)  # LUT from MCU
        self.epd_w21_write_data(0x0D)

        # Power Setting
        self.epd_w21_write_cmd(0x01)
        self.epd_w21_write_data(0x03)  # Enable internal VSH, VSL, VGH, VGL
        self.epd_w21_write_data(self.LUT_ALL[211])  # VGH=20V, VGL=-20V
        self.epd_w21_write_data(self.LUT_ALL[212])  # VSH=15V
        self.epd_w21_write_data(self.LUT_ALL[213])  # VSL=-15V
        self.epd_w21_write_data(self.LUT_ALL[214])  # VSHR

        # Booster Soft Start
        self.epd_w21_write_cmd(0x06)
        self.epd_w21_write_data(0xD7)  # D7
        self.epd_w21_write_data(0xD7)  # D7
        self.epd_w21_write_data(0x27)  # 2F

        # PLL Control - Frame Rate
        self.epd_w21_write_cmd(0x30)
        self.epd_w21_write_data(self.LUT_ALL[210])  # PLL

        # CDI Setting
        self.epd_w21_write_cmd(0x50)
        self.epd_w21_write_data(0x57)

        # TCON Setting
        self.epd_w21_write_cmd(0x60)
        self.epd_w21_write_data(0x22)

        # Resolution Setting
        self.epd_w21_write_cmd(0x61)
        self.epd_w21_write_data(0xF0)  # HRES[7:3] - 240
        self.epd_w21_write_data(0x01)  # VRES[15:8] - 320
        self.epd_w21_write_data(0xA0)  # VRES[7:0]

        self.epd_w21_write_cmd(0x65)
        self.epd_w21_write_data(0x00)  # Additional resolution setting, if needed

        # VCOM_DC Setting
        self.epd_w21_write_cmd(0x82)
        self.epd_w21_write_data(self.LUT_ALL[215])  # -2.0V

        # Power Saving Register
        self.epd_w21_write_cmd(0xE3)
        self.epd_w21_write_data(0x88)  # VCOM_W[3:0], SD_W[3:0]

        # LUT Setting
        self.write_full_lut()

        # Power ON
        self.epd_w21_write_cmd(0x04)
        self.lcd_chkstatus()  # Check if the display is ready

    def pic_display_4g(self,datas):
        # Command to start transmitting old data
        buffer = []
        self.epd_w21_write_cmd(0x10)
    
        self.GPIO.output(self.DC_PIN, GPIO.HIGH)  # Data mode

        print("Start Old Data Transmission")
        # Iterate over each byte of the image data
        for i in range(12480):  # Assuming 416x240 resolution, adjust accordingly
            temp3 = 0
            for j in range(2):  # For each half-byte in the data
                temp1 = datas[i * 2 + j]
                for k in range(4):  # For each bit in the half-byte
                    temp2 = temp1 & 0xC0
                    if temp2 == 0xC0:
                        temp3 |= 0x01  # White
                    elif temp2 == 0x00:
                        temp3 |= 0x00  # Black
                    elif temp2 == 0x80:
                        temp3 |= 0x01  # Gray1
                    elif temp2 == 0x40:
                        temp3 |= 0x00  # Gray2

                    if j==0:
                        temp1 <<= 2
                        temp3 <<= 1
                    if j==1 and k != 3:
                        temp1 <<= 2
                        temp3 <<= 1
            buffer.append(temp3)
        self.spi.xfer3(buffer, self.spi.max_speed_hz, 1 ,8)

        buffer = []
        print("Start New Data Transmission")
        # Command to start transmitting new data
        self.epd_w21_write_cmd(0x13)
        self.GPIO.output(self.DC_PIN, GPIO.HIGH)  # Data mode

        for i in range(12480):  # Repeat the process for new data
            temp3 = 0
            for j in range(2):
                temp1 = datas[i * 2 + j]
                for k in range(4):
                    temp2 = temp1 & 0xC0
                    # The logic for determining color values remains the same
                    if temp2 == 0xC0:
                        temp3 |= 0x01  # White
                    elif temp2 == 0x00:
                        temp3 |= 0x00  # Black
                    elif temp2 == 0x80:
                        temp3 |= 0x00  # Gray1
                    elif temp2 == 0x40:
                        temp3 |= 0x01  # Gray2
                
                    if j==0:
                        temp1 <<= 2
                        temp3 <<= 1
                    if j==1 and k != 3:
                        temp1 <<= 2
                        temp3 <<= 1
            buffer.append(temp3)

        self.spi.xfer3(buffer, self.spi.max_speed_hz, 1 ,8)

        # Refresh command
        print("Refreshing")
        self.epd_w21_write_cmd(0x12)
        self.delay_xms(1)  # Necessary delay for the display refresh
        self.lcd_chkstatus()  # Check the display status

    def PIC_display(self,new_data):
        # Assuming oldData is globally defined or accessible
        
        # Transfer old data
        self.epd_w21_write_cmd(0x10)
        self.GPIO.output(self.DC_PIN, GPIO.HIGH)  # Data mode
        self.spi.xfer3(self.oldData, self.spi.max_speed_hz, 1 ,8)

        # Transfer new data
        self.epd_w21_write_cmd(0x13)
        self.GPIO.output(self.DC_PIN, GPIO.HIGH)  # Data mode
        self.spi.xfer3(new_data, self.spi.max_speed_hz, 1 ,8)
        self.oldData = new_data.copy()
        
        # Refresh display
        self.epd_w21_write_cmd(0x12)
        self.delay_xms(1)  # Necessary delay for the display refresh
        self.lcd_chkstatus()  # Check if the display is ready

    def PIC_display_Clear(self,poweroff=False):
        # Transfer old data
        self.epd_w21_write_cmd(0x10)
        self.GPIO.output(self.DC_PIN, GPIO.HIGH)  # Data mode
        self.spi.xfer3(self.oldData, self.spi.max_speed_hz, 1 ,8)
        
        # Transfer new data, setting all to 0xFF (white or clear)
        self.epd_w21_write_cmd(0x13)
        self.GPIO.output(self.DC_PIN, GPIO.HIGH)  # Data mode
        self.spi.xfer3([0] * 12480, self.spi.max_speed_hz, 1 ,8)
        self.oldData = [0] * 12480

        
        # Refresh the display
        self.epd_w21_write_cmd(0x12)
        self.delay_xms(1)  # Ensure a small delay for the display to process
        self.lcd_chkstatus()  # Check the display status

        if poweroff:
            self.power_off()  # Optionally power off the display after clearing

#Instruction Set
EncodeTable = {"BTN_UP": 0b1, "BTN_DOWN": 0b10, "BTN_SELECT": 0b100}

# Set up GPIO pins
selectBTN = machine.Pin(16, machine.Pin.IN, machine.Pin.PULL_DOWN)
upBTN = machine.Pin(17, machine.Pin.IN, machine.Pin.PULL_DOWN)
downBTN = machine.Pin(18, machine.Pin.IN, machine.Pin.PULL_DOWN)
powerStatus = machine.Pin(21, machine.Pin.OUT)
einkStatus = machine.Pin(9, machine.Pin.OUT)

# Setup UART0 on GPIO0 (TX) and GPIO1 (RX)
uart0 = machine.UART(0, baudrate=9600, tx=machine.Pin(0), rx=machine.Pin(1))

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
    uart0.write(f"{state_byte}\n")

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
            einkStatus.high()  # inverted logic
            uart0.write("xPOWER_ON\n")

    if debounce(upBTN) and selectBTN.value() == 1:
        uart0.write("xUP\n")
        start_time = utime.ticks_ms()
        while utime.ticks_diff(utime.ticks_ms(), start_time) < 5000:
            if upBTN.value() == 0 or selectBTN.value() == 0:
                break
            utime.sleep_ms(10)
        if utime.ticks_diff(utime.ticks_ms(), start_time) >= 5000 and powerStatus.value() == 1:
            powerStatus.low()
            einkStatus.low()  # inverted logic
            uart0.write("xPOWER_OFF\n")
    utime.sleep_ms(1)






