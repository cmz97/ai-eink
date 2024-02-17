
import time
import spidev
import RPi.GPIO as GPIO
import numpy as np

LUT_ALL = [
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
emptyImage = [0xFF] * 24960

# # Load the .npy file without immediately converting to integers
# image_4g_strings = np.load('/home/kevin/Desktop/image_data.npy')

# # Convert each hexadecimal string to an unsigned 8-bit integer
# image_4g = np.array([int(x, 16) for x in image_4g_strings.flatten()], dtype=np.uint8).reshape(image_4g_strings.shape)

#Pin Def
DC_PIN = 6
RST_PIN = 13
BUSY_PIN = 19

def EPD_GPIO_Init():
    # GPIO.cleanup()
    GPIO.setwarnings(False) 
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(DC_PIN, GPIO.OUT) #DC 
    GPIO.setup(RST_PIN, GPIO.OUT) #REST
    GPIO.setup(BUSY_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP) #BUSY


    bus = 0 # We only have SPI bus 0 available to us on the Pi
    device = 0 #Device is the chip select pin. Set to 0 or 1, depending on the connections
    spi = spidev.SpiDev()
    spi.open(bus, device) 
    spi.max_speed_hz = 1000000 #1MHZ
    spi.mode = 0

    return spi

def SPI_Delay():
    time.sleep(0.001)

def SPI_Write(value):
    return spi.xfer2([value])

def epd_w21_write_cmd(command):
    SPI_Delay()
    GPIO.output(DC_PIN, GPIO.LOW)  # Command mode
    SPI_Write(command)

def epd_w21_write_data(data):
    SPI_Delay()
    GPIO.output(DC_PIN, GPIO.HIGH)  # Data mode
    SPI_Write(data)

def delay_xms(xms):
    time.sleep(xms / 1000.0)

def epd_w21_init():
    delay_xms(100)  # At least 10ms delay
    GPIO.output(RST_PIN, False)  # Module reset
    delay_xms(20)
    GPIO.output(RST_PIN, True)
    delay_xms(20)

EPD_WIDTH = 240 
EPD_HEIGHT = 416 

def EPD_Display(image):
    width = (EPD_WIDTH + 7) // 8
    height = EPD_HEIGHT

    epd_w21_write_cmd(0x10)
    for j in range(height):
        for i in range(width):
            epd_w21_write_data(image[i + j * width])

    epd_w21_write_cmd(0x13)
    for _ in range(height * width):
        epd_w21_write_data(0x00) 

    epd_w21_write_cmd(0x12)
    delay_xms(1)  # Necessary delay
    # You would need to implement lcd_chkstatus here

def lcd_chkstatus():
    while GPIO.input(BUSY_PIN) == GPIO.LOW:  # Assuming LOW means busy
        time.sleep(0.01)  # Wait 10ms before checking again
        print("BUSY...")

def epd_sleep():
    epd_w21_write_cmd(0x02)  # Power off
    lcd_chkstatus()  # Implement this to check the display's busy status
    
    epd_w21_write_cmd(0x07)  # Deep sleep
    epd_w21_write_data(0xA5)

def epd_init():
    epd_w21_init()  # Reset the e-paper display
    
    epd_w21_write_cmd(0x04)  # Power on
    lcd_chkstatus()  # Implement this to check the display's busy status

    epd_w21_write_cmd(0x50)  # VCOM and data interval setting
    epd_w21_write_data(0x97)  # Settings for your display

def epd_init_fast():
    epd_w21_init()  # Reset the e-paper display
    
    epd_w21_write_cmd(0x04)  # Power on
    lcd_chkstatus()  # Implement this to check the display's busy status

    epd_w21_write_cmd(0xE0)
    epd_w21_write_data(0x02)

    epd_w21_write_cmd(0xE5)
    epd_w21_write_data(0x5A)

def epd_init_part():
    epd_w21_init()  # Reset the e-paper display
    
    epd_w21_write_cmd(0x04)  # Power on
    lcd_chkstatus()  # Implement this to check the display's busy status

    epd_w21_write_cmd(0xE0)
    epd_w21_write_data(0x02)

    epd_w21_write_cmd(0xE5)
    epd_w21_write_data(0x6E)

    epd_w21_write_cmd(0x50)
    epd_w21_write_data(0xD7)

def power_off():
    epd_w21_write_cmd(0x02)
    lcd_chkstatus()

def write_full_lut():
    epd_w21_write_cmd(0x20)  # Write VCOM register
    for i in range(42):
        epd_w21_write_data(LUT_ALL[i])

    epd_w21_write_cmd(0x21)  # Write LUTWW register
    for i in range(42, 84):
        epd_w21_write_data(LUT_ALL[i])

    epd_w21_write_cmd(0x22)  # Write LUTR register
    for i in range(84, 126):
        epd_w21_write_data(LUT_ALL[i])

    epd_w21_write_cmd(0x23)  # Write LUTW register
    for i in range(126, 168):
        epd_w21_write_data(LUT_ALL[i])

    epd_w21_write_cmd(0x24)  # Write LUTB register
    for i in range(168, 210):
        epd_w21_write_data(LUT_ALL[i])

def epd_w21_init_4g():
    epd_w21_init()  # Reset the e-paper display

    # Panel Setting
    epd_w21_write_cmd(0x00)
    epd_w21_write_data(0xFF)  # LUT from MCU
    epd_w21_write_data(0x0D)

    # Power Setting
    epd_w21_write_cmd(0x01)
    epd_w21_write_data(0x03)  # Enable internal VSH, VSL, VGH, VGL
    epd_w21_write_data(LUT_ALL[211])  # VGH=20V, VGL=-20V
    epd_w21_write_data(LUT_ALL[212])  # VSH=15V
    epd_w21_write_data(LUT_ALL[213])  # VSL=-15V
    epd_w21_write_data(LUT_ALL[214])  # VSHR

    # Booster Soft Start
    epd_w21_write_cmd(0x06)
    epd_w21_write_data(0xD7)  # D7
    epd_w21_write_data(0xD7)  # D7
    epd_w21_write_data(0x27)  # 2F

    # PLL Control - Frame Rate
    epd_w21_write_cmd(0x30)
    epd_w21_write_data(LUT_ALL[210])  # PLL

    # CDI Setting
    epd_w21_write_cmd(0x50)
    epd_w21_write_data(0x57)

    # TCON Setting
    epd_w21_write_cmd(0x60)
    epd_w21_write_data(0x22)

    # Resolution Setting
    epd_w21_write_cmd(0x61)
    epd_w21_write_data(0xF0)  # HRES[7:3] - 240
    epd_w21_write_data(0x01)  # VRES[15:8] - 320
    epd_w21_write_data(0xA0)  # VRES[7:0]

    epd_w21_write_cmd(0x65)
    epd_w21_write_data(0x00)  # Additional resolution setting, if needed

    # VCOM_DC Setting
    epd_w21_write_cmd(0x82)
    epd_w21_write_data(LUT_ALL[215])  # -2.0V

    # Power Saving Register
    epd_w21_write_cmd(0xE3)
    epd_w21_write_data(0x88)  # VCOM_W[3:0], SD_W[3:0]

    # LUT Setting
    write_full_lut()

    # Power ON
    epd_w21_write_cmd(0x04)
    lcd_chkstatus()  # Check if the display is ready

def pic_display_4g(datas):
    # Command to start transmitting old data
    epd_w21_write_cmd(0x10)
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
        epd_w21_write_data(temp3)

    print("Start New Data Transmission")
    # Command to start transmitting new data
    epd_w21_write_cmd(0x13)
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
        epd_w21_write_data(temp3)

    # Refresh command
    print("Refreshing")
    epd_w21_write_cmd(0x12)
    delay_xms(1)  # Necessary delay for the display refresh
    lcd_chkstatus()  # Check the display status


# spi = EPD_GPIO_Init()

# print("Started")
# epd_w21_init_4g()
# print("Finished Init")
# pic_display_4g(image_4g)
# epd_sleep()	
# time.sleep(2)
# print("Image Displayed")
