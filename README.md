# ai-eink

force_turbo=1
NFC Driver: https://github.com/2pecshy/eeprom-ST25DV-linux-driver
sudo apt install -y python3-libcamera python3-kms++ libcap-dev
 pip3 install picamera2 --break-system-packages

# Drivers
## Camera 
sudo wget https://datasheets.raspberrypi.org/cmio/dt-blob-cam1.bin -O /boot/dt-blob.bin

libcamera-still -q 80 -o test.jpg

For OS Lite 
Open Terminal and enter "sudo nano /boot/config.txt"
Scroll down to the line that reads "camera_auto_detect=1" and change it to "camera_auto_detect=0"
On the next line, enter "dtoverlay=imx219"
Reboot


## Speakers
https://github.com/HinTak/seeed-voicecard
sudo aplay -Dhw:0 test.wav
https://www.waveshare.com/wiki/WM8960_Audio_HAT
sudo alsactl store

## Pi-PICO
ampy --port /dev/tty.usbmodem1112401 put main.py

## Radxa
sudo apt-get install python3-dev spidev
sudo apt-get update
sudo apt-get install python3-libgpiod