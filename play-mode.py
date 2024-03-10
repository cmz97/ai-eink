import os
import sys
import time 
import random
import json
from einkDSP import einkDSP
from encoder import Encoder, Button
from PIL import Image
from utils import * 
import threading  # Import threading module
import RPi.GPIO as GPIO
from apps import SdBaker

class Controller:

    def __init__(self, rotCallback, butCallback):
        self.eink = einkDSP()
        self.rot = Encoder(22, 17, callback=rotCallback) # 22 17
        self.but = Button(26, callback=butCallback) # gpio 26

    def transit(self):
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()

    def part_screen(self, hex_pixels):
        self.eink.epd_init_part()
        self.eink.PIC_display(hex_pixels)
        
    def full_screen(self, hex_pixels):
        self.eink.epd_w21_init_4g()
        self.eink.pic_display_4g(hex_pixels)
        self.eink.epd_sleep()

    def image_callback(self, image, prompt):
        dialogBox = draw_text_on_dialog(prompt)
        post_img = process_image(image, dialogBox)
        hex_pixels = image_to_header_file(post_img)
        self.full_screen(hex_pixels)
        

def rotCallback(counter, direction):
    pass

def butCallback():
    pass

if __name__ == "__main__":
    controller = Controller(rotCallback, butCallback)
    sd_baker = SdBaker()
    init_pormpt = sd_baker.get_prompt()
    # init first display
    event = sd_baker.generate_image(add_prompt=init_pormpt, callback=controller.image_callback)
    try:
        while True:
            time.sleep(3)
            print(f"ping - \n")
    except Exception:
        pass

    GPIO.cleanup()
