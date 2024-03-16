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
from apps import SdBaker, PromptsBank
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Controller:

    def __init__(self):
        self.eink = einkDSP()
        self.rot = Encoder(22, 17, callback=self.rotCallback) # 22 17
        self.but = Button(26, callback=self.butCallback) # gpio 26
        self.in_4g = True
        self.image = None
        
        # UI states
        self.page = 0 
        self.pending_swap_prompt = None
        self.current_screen_selects = []
        self.layout = {
            0 : self._butAction0,
            1 : self._butAction1
        }
        self.selection_idx = [0] * len(self.layout)
        logging.info('Controller instance created')

        # threading issues
        self.locked = False
        self.loading_thread = None
        self.stop_loading_event = threading.Event()

    def show_loading_screen(self):
        start_time = time.time()
        frame0 = paste_loadingBox(self.image, frame=0)
        frame1 = paste_loadingBox(self.image, frame=1)
            
        while not self.stop_loading_event.is_set():
            time.sleep(0.5)
            draw_text_on_img("{:.0f}s".format(time.time() - start_time), frame0)
            pixels = dump_2bit(np.array(frame0.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32)).tolist()
            self.part_screen(pixels)
            time.sleep(0.5)
            draw_text_on_img("{:.0f}s".format(time.time() - start_time), frame1)
            pixels = dump_2bit(np.array(frame1.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32)).tolist()
            self.part_screen(pixels)

    def transit(self):
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()
        logging.info('transit to 2g done')

    def part_screen(self, hex_pixels):
        self.eink.epd_init_part()
        self.eink.PIC_display(hex_pixels)
        
    def full_screen(self, hex_pixels):
        self.eink.epd_w21_init_4g()
        self.eink.pic_display_4g(hex_pixels)
        self.eink.epd_sleep()

    def _status_check(self):
        if self.in_4g : 
            self.transit()
            self.in_4g = False
        
    def rotCallback(self, counter, direction):    
        # sanity check 
        if self.locked : return False
        self._status_check()
        logging.info("* Direction: {}".format(direction))

        # select prompt to edit
        self.selection_idx[self.page] += -1 if direction == "U" else 1
        self.selection_idx[self.page] = self.selection_idx[self.page] % len(self.current_screen_selects)


        # update screen
        self.update_screen()


    def update_screen(self):
        # prepare menu
        updated_img = override_dialogBox(self.image, self._prepare_menu())

        # update screen
        grayscale = updated_img.transpose(Image.FLIP_TOP_BOTTOM).convert('L')
        pixels = np.array(grayscale, dtype=np.float32)
        logging.info('preprocess image done')
        hex_pixels = dump_2bit(pixels).tolist()
        logging.info('2bit pixels dump done')
        self.part_screen(hex_pixels)


    def _butAction0(self):
        # update current select and next screen
        self.pending_swap_prompt = self.current_screen_selects[self.selection_idx[self.page]]
        self.current_screen_selects = pb.get_candidates(self.pending_swap_prompt)

        # update vars
        self.page = 1 # enter selection page

        # update and fresh new screen selects
        self.update_screen()



    def _butAction1(self):
        # submit, update vars
        pb.update_prompt(self.pending_swap_prompt, self.current_screen_selects[self.selection_idx[self.page]])
        self.page = 0 # back to prompt list page
        
        # update screen
        self.current_screen_selects = pb.prompt
        self.update_screen()

    def _butSubmit(self):
        # resets
        self.selection_idx = [0] * len(self.layout)
        self.page = 0

        if not self.pending_swap_prompt:
            self.image_callback(self.image, False)
            return

        self.sd_process()

    def sd_process(self, prompts=None):
        if not prompts: prompts = pb.to_str() 
        self.locked = True

        # Start a new thread for the loading screen
        self.stop_loading_event.clear()
        self.loading_thread = threading.Thread(target=self.show_loading_screen)
        self.loading_thread.start()
        # prepare prompt and trigger gen
        event = sd_baker.generate_image(add_prompt=prompts, callback=self.image_callback)
        event.wait()

    def butCallback(self, option):
        # sanity check 
        if self.locked : return False
        self._status_check()
        
        if option == 0 : self.layout[self.page]()
        if option == 1 : self._butSubmit()

    def _prepare_menu(self):
        text = ""
        for _, option in enumerate(self.current_screen_selects):
            text += f"{option}\n"
        dialogBox = draw_text_on_dialog(text, highlighted_lines=[self.selection_idx[self.page]])
        return dialogBox

    def image_callback(self, image, update=True):

        if update:
            # cache current prompts 
            self.current_screen_selects = pb.prompt
            # new prompts selection
            pb.fresh_prompt_selects()
            logging.info('fresh prompt selects done')
            # update image
            post_img = process_image(image, self._prepare_menu())
            hex_pixels = image_to_header_file(post_img)
        else:
            hex_pixels = image_to_header_file(image)

        # handle threadings
        self.stop_loading_event.set()
        if self.loading_thread : 
            self.loading_thread.join()
            self.loading_thread = None 
        time.sleep(0.5)    
        
        # show and update states
        self.full_screen(hex_pixels)
        self.in_4g = True
        self.image = post_img
        self.locked = False
        

# init 
pb = PromptsBank()
controller = Controller()
file_cache = "./temp.png"

if __name__ == "__main__":
    sd_baker = SdBaker(pb)
    # init first display
    if os.path.exists(file_cache):
        image = Image.open(file_cache)
        prompt_str = image.info.get("prompt")
        pb.load_prompt(prompt_str)  
        controller.image_callback(image)
    else:
        controller.sd_process()
        
    try:
        while True:
            time.sleep(3)
            print(f"ping - \n")
    except Exception:
        pass

    GPIO.cleanup()
