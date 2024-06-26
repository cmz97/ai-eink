# from sshkeyboard import listen_keyboard
import os
import sys
import time 
import random
import json
from pathlib import Path
from einkDSP import einkDSP
from encoder import Encoder, Button, MultiButtonMonitor
from PIL import Image
from utils import * 
import threading  # Import threading module
import RPi.GPIO as GPIO
from apps import SdBaker, PromptsBank, BookLoader, SceneBaker
import logging
from Ebook_GUI import EbookGUI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
GPIO.cleanup()

class Controller:

    def __init__(self):
        self.eink = einkDSP()

        # self.butUp = Button(9, direction='up', callback=self.press_callback) # gpio 26
        # self.butDown = Button(22, direction='down', callback=self.press_callback) # gpio 26
        # self.butEnter = Button(17, direction='enter', callback=self.press_callback) # gpio 26

        buttons = [
            {'pin': 9, 'direction': 'up', 'callback': self.press_callback},
            {'pin': 22, 'direction': 'down', 'callback': self.press_callback},
            {'pin': 17, 'direction': 'enter', 'callback': self.press_callback}
        ]
        
        self.multi_button_monitor = MultiButtonMonitor(buttons)

        self.in_4g = True
        self.image = Image.new("L", (eink_width, eink_height), "white")
        self.model = None
        
        # UI states
        self.page = 0 # 0 : book read 
        self.pending_swap_prompt = None
        self.current_screen_selects = []
        self.layout = {
            0 : self._select_book,
            1 : self._read_page,
        }
        self.selection_idx = [0] * len(self.layout)
        # self.display_cache = {
        #     1 : [text_to_image("[edit prompts]"), text_to_image("[generate]"), text_to_image("[back]")],
        #     2 : [text_to_image(f"{group}") for group in pb.prompt_groups] + [text_to_image("[back]")],
        #     3 : [] # pending to be init
        # }
        logging.info('Controller instance created')

        # buffers
        self.text_buffer = ["", ]
        self.image_buffer = []

        # threading issues
        self.locked = False
        self.cooking = False
        self.pending_image = False
        self.image_preview_page = None

        
    def background_task(self):
        # llm
        # self._fast_text_display("llm ...")
        # self._fast_text_display("llm processing ~40sec")
        prompt = sb.get_next_scene(" ".join(self.text_buffer[-1]))
        # self._fast_refresh()
        # sd gen
        # self._fast_text_display("sd ...")
        self.sd_process(prompt)
        # self._fast_refresh()

    def transit(self):
        self.locked = True
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()
        logging.info('transit to 2g done')
    
    def clear_screen(self):
        image = Image.new("L", (eink_width, eink_height), "white")
        hex_pixels = dump_1bit(np.array(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.uint8))
        self.eink.epd_init_part()
        self.eink.PIC_display(hex_pixels)


    def part_screen(self, hex_pixels):
        self.locked = True
        self.eink.epd_init_part()
        self.eink.PIC_display(hex_pixels)
        self.locked = False
        
    def full_screen(self, hex_pixels):
        self.eink.epd_w21_init_4g()
        self.eink.pic_display_4g(hex_pixels)
        self.eink.epd_sleep()

    def _status_check(self):
        if self.in_4g : 
            self.transit()
            self.in_4g = False
    
    def _fast_refresh(self):
        pixels = dump_1bit_with_dithering(np.array(self.image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32))
        self.part_screen(pixels)

    def _fast_text_display(self, text="loading ..."):
        image = fast_text_display(self.image, text)
        grayscale = image.transpose(Image.FLIP_TOP_BOTTOM).convert('L')
        hex_pixels = dump_1bit_with_dithering(np.array(grayscale, dtype=np.float32))
        # hex_pixels = dump_1bit(np.array(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.uint8))
        self.part_screen(hex_pixels)

    def update_screen(self):
        image = self.image
        # image = self._prepare_menu(self.image)
        # update screen
        grayscale = image.transpose(Image.FLIP_TOP_BOTTOM).convert('L')
        logging.info('preprocess image done')
        hex_pixels = dump_1bit_with_dithering(np.array(grayscale, dtype=np.float32))
        logging.info('2bit pixels dump done')
        self.part_screen(hex_pixels)

    def load_model(self):
        logger.info("loading model")
        self._fast_text_display("loading sd model ~40sec")
        sd_baker.load_model(
            '/home/kevin/ai/models/sdxs-512-dreamshaper-onnx',
            "sdxs",
            "")
        self._fast_text_display("sd model loaded!")
        # self._fast_refresh()
    
    def _select_book(self, key):
        self.locked = True
        # sync status
        self.page = 0
        curr_file = str(model_list[self.selection_idx[self.page]])
        ebk.loadFile(curr_file)

        # process key
        if key == "up" : self.selection_idx[self.page] += 1 
        elif key == "down" : self.selection_idx[self.page] -= 1 
        elif key == "enter" :  # load book
            self._read_page("init")
            return

        # rolling        
        self.selection_idx[self.page] = self.selection_idx[self.page] % len(model_list)

        self.clear_screen()
        # print screen
        thumbnail_image = render_thumbnail_page(Image.open("/".join(curr_file.split("/")[0:-1])+"/thumbnail.png"), "")        
        self.image = thumbnail_image
        # hex_pixels = image_to_header_file(thumbnail_image)
        grayscale = thumbnail_image.transpose(Image.FLIP_TOP_BOTTOM).convert('L')
        hex_pixels = dump_1bit_with_dithering(np.array(grayscale, dtype=np.float32))
        self.part_screen(hex_pixels) 
        self.locked = False


    def _read_page(self, key): # main page
        self.locked = True
        # sync status
        self.page = 1
        # process key
        if key == "up" : self.selection_idx[self.page] -= 1 
        elif key == "down" : self.selection_idx[self.page] += 1 

        logger.info(f"status self.image_preview_page {self.image_preview_page}, self.selection_idx[self.page] {self.selection_idx[self.page]}")

        if self.image_preview_page and self.image_preview_page == self.selection_idx[self.page] : 
            # show image instead
            self.show_last_image()
            self.selection_idx[self.page] = self.selection_idx[self.page] - 1
            self.image_preview_page = None
            return

        self._status_check()

        # rolling        
        self.selection_idx[self.page] = max(1, self.selection_idx[self.page]) # % len(self.display_cache[self.page])

        # loading resources
        # text_to_display = ebk.getPage(self.selection_idx[self.page])
        if self.selection_idx[self.page] >= len(self.text_buffer):
            self.text_buffer.append(ebk.getPage(self.selection_idx[self.page]))
        text_to_display = self.text_buffer[self.selection_idx[self.page]]
        
        # pre send to sd
        if self.selection_idx[self.page] == len(self.text_buffer) - 1: 
            logger.info("pending image flag up")
            self.pending_image = True
            self.text_buffer.append(ebk.getPage(self.selection_idx[self.page] + 1)) # next

        # logger.info(text_to_display)
        # images 
        # line_img = [text_to_image(line) for line in text_to_display]
        gui.clear_page()
        self.image = gui.draw_text_on_canvas(gui.canvas, ["".join(x) for x in text_to_display])
    
        # self.image = draw_text_on_screen(line_img)
        # print screen
        # hex_pixels = image_to_header_file(image)
        # self.full_screen(hex_pixels)
        self.update_screen()        
        self.locked = False 

    def trigger_background_job(self):
        logger.info("background_task triggered")
        background_thread = threading.Thread(target=self.background_task, daemon=True)
        background_thread.start()
        # update flag
        self.pending_image = False
        logger.info("pending image flag off")

    def sd_process(self, prompts=None):
        if not prompts: prompts = pb.to_str() 
        # Start a new thread for the loading screen
        # prepare prompt and trigger gen
        # event = sd_baker.generate_image(add_prompt=prompts, callback=self.sd_image_callback)
        # event.wait()
        sd_baker._generate_image_thread(prompts, self.sd_image_callback, "temp-ebook")


    def press_callback(self, key):
        if self.locked : return
        print(f"Key {key} pressed.")
        if key == "q":
            print("Quitting...")
            exit()
        
        self.layout[self.page](key)
        
    def sd_image_callback(self, image):
        # post_img = process_image(image)
        # image_with_dialog = self._prepare_menu(post_img)
        self.image_buffer.append(image)
        self.image_preview_page = self.selection_idx[self.page] + 2

        # image finished cooking done
        self.cooking = False

    def show_last_image(self):
        # add some details
        image = insert_image(self.image, self.image_buffer.pop(0))
        hex_pixels = image_to_header_file(image)
        # show and update states
        self.clear_screen()
        self.full_screen(hex_pixels)
        self.in_4g = True
        self.locked = False
    
        
# book folder
model_list = ['/home/kevin/ai/books/dune/Dune.txt']

# init 
# pb = PromptsBank()
sb = SceneBaker("Dune")
sd_baker = SdBaker(vae_override="../models/sdxs-512-0.9/vae")
# override
sd_baker.neg_prompt = "ng_deepnegative_v1_75t, bad hand, bad face, worst quality, low quality, logo, text, watermark, username, harsh shadow, shadow, artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate"
sd_baker.char_id = "graphic novel, dune movie,"
sd_baker.num_inference_steps = 1

# scale with gui box
gui = EbookGUI()
w, h = gui.text_area[1][0] - gui.text_area[0][0] , gui.text_area[1][1] - gui.text_area[0][1]
scale = 0.55
ebk = BookLoader(filePath='../models/Dune.txt', 
screenWidth=w,
screenHeight=h,
fontWidth=gui.font_size * scale, 
fontHeight=gui.font_size * scale * 0.95)

controller = Controller()
file_cache = "./temp.png"
backCounter = 0

if __name__ == "__main__":
    # sd_baker = SdBaker(pb)
    # start with main screen

    # all the preheats
    controller.layout[0]('init')
    controller.load_model()
    controller.text_buffer.append(ebk.getPage(1))
    controller.pending_image = True

    try:
        while True:
            time.sleep(1)
            print(f"ping - \n")
            if controller.pending_image and not controller.cooking:
                # trigger sd cooking tasks
                logger.info("background tasks triggerd")
                controller.cooking = True
                controller.trigger_background_job()
            # backCounter += 1 if GPIO.input(9) == 1 else 0
            # if backCounter >= 5:
            #     os._exit(0)
                

    except Exception:
        pass

    GPIO.cleanup()

    