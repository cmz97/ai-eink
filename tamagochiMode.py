# from sshkeyboard import listen_keyboard
import os
import sys
import time 
import random
import json
from pathlib import Path
from einkDSP import einkDSP
from encoder import Encoder, Button
from PIL import Image
from utils import * 
import threading  # Import threading module
import RPi.GPIO as GPIO
from apps import SdBaker, PromptsBank, BookLoader, SceneBaker
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
GPIO.cleanup()

class Controller:

    def __init__(self):
        self.eink = einkDSP()
        self.butUp = Button(9, direction='up', callback=self.press_callback) # gpio 26
        self.butDown = Button(22, direction='down', callback=self.press_callback) # gpio 26
        self.butEnter = Button(17, direction='enter', callback=self.press_callback) # gpio 26

        self.in_4g = True
        self.image = Image.new("L", (eink_width, eink_height), "white")
        self.model = None
        
        # UI states
        self.page = 0 # 0 : status display, 1 : image display
        self.pending_swap_prompt = None
        self.current_screen_selects = []
        self.layout = {
            0 : self._show_status,
            1 : self._image_display,
        }
        self.selection_idx = [0] * len(self.layout)
        self.display_cache = {
            1 : [text_to_image("Waifu doing ok  ⸜(｡ ˃ ᵕ ˂ )⸝♡")],
            2 : [text_to_image("[save]")] + [text_to_image("[back]")],
        }
        logging.info('Controller instance created')

        # buffers
        self.text_buffer = ["", ]
        self.image_buffer = []
        self.action_buffer = []

        # threading issues
        self.locked = False
        self.animation_thread = None
        self.stop_animation_event = threading.Event()

        self.cooking = False
        self.pending_image = False
        self.image_preview_page = None

        


    def background_task(self):
        # llm
        # prompt = sb.get_next_scene(" ".join(self.text_buffer[-1]))
        # sd gen
        # TODO change prompt given state
        self.sd_process("")



    # EINK DISPLAY
    def transit(self):
        self.locked = True
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()
        logging.info('transit to 2g done')
    
    def clear_screen(self):
        self.eink.PIC_display_Clear()
        
    def part_screen(self, hex_pixels):
        self.locked = True
        self.eink.epd_init_part()
        self.eink.PIC_display(hex_pixels)
        self.locked = False
        
    def full_screen(self, hex_pixels):
        self.eink.epd_w21_init_4g()
        self.eink.pic_display_4g(hex_pixels)
        self.eink.epd_sleep()

    
    def update_screen(self):
        image = self.image
        # image = self._prepare_menu(self.image)
        # update screen
        grayscale = image.transpose(Image.FLIP_TOP_BOTTOM).convert('L')
        pixels = np.array(grayscale, dtype=np.float32)
        logging.info('preprocess image done')
        hex_pixels = dump_2bit(pixels).tolist()
        logging.info('2bit pixels dump done')
        self.part_screen(hex_pixels)



    # UTILS
    def _status_check(self):
        if self.in_4g : 
            self.transit()
            self.in_4g = False
            

    def load_model(self):
        logger.info("loading model")
        model_path = model_list[0].replace("_add_ons.json","")
        with open(model_list[0]) as f: model_info = json.load(f)
        sd_baker.load_model(model_path, model_info['name'], model_info['trigger_words'])


    def _prepare_menu(self, image):     
        if self.page == 0:
            image_with_dialog = apply_dialog_box(
                input_image = image,
                dialog_image = ui_assets["large_dialog_box"]["image"],
                box_mat = self.display_cache[self.page],
                highligh_index = self.selection_idx[self.page],
                placement_pos = ui_assets["large_dialog_box"]["placement_pos"]
            )
        elif self.page == 1:
            sub_page_id = self.selection_idx[self.page-1] # get the cache based on previous page selection
            image_with_dialog = apply_dialog_box(
                input_image = image,
                dialog_image = ui_assets["small_dialog_box"]["image"],
                box_mat = self.display_cache[self.page][sub_page_id],
                highligh_index = self.selection_idx[self.page],
                placement_pos = ui_assets["small_dialog_box"]["placement_pos"]
            )
        return image_with_dialog


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
        event = sd_baker.generate_image(add_prompt=prompts, callback=self.sd_image_callback)
        event.wait()

    # STATE MACHINE
    def start_animation(self):
        self.locked = True
        logging.info("start_loading_screen thread ")
        self.stop_animation_event.clear()
        if self.animation_thread is None or not self.animation_thread.is_alive():
            self.animation_thread = threading.Thread(target=self.show_animation)
            self.animation_thread.start()

    def stop_start_animation(self):
        logging.info("stop_loading_screen thread ")
        self.stop_animation_event.set()
        if self.animation_thread:
            self.animation_thread.join()
            self.animation_thread = None

    def show_animation(self):
        self.locked = True
        start_time = time.time()

        if self.action_buffer == []:
            frame0 = paste_loadingBox(self.image, frame=0)
            frame1 = paste_loadingBox(self.image, frame=1)
            # frame0 = draw_text_on_dialog("COOKING...", frame0, (eink_width//2, eink_height//3*2), (eink_width//2+50, eink_height//3*2), True)
            # frame1 = draw_text_on_dialog("COOKING...", frame1, (eink_width//2, eink_height//3*2), (eink_width//2+50, eink_height//3*2), True)
            while not self.stop_animation_event.is_set():
                draw_text_on_img("{:.0f}s".format(time.time() - start_time), frame0)            
                pixels = dump_2bit(np.array(frame0.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32)).tolist()
                self.part_screen(pixels)
                time.sleep(0.5)
                draw_text_on_img("{:.0f}s".format(time.time() - start_time), frame1)
                pixels = dump_2bit(np.array(frame1.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32)).tolist()
                self.part_screen(pixels)
                time.sleep(0.5)

        self.locked = False


    def _show_status(self, actions=[]):
        self.page = 0
        self._status_check()

        if not actions: 
            # prepare image and dialog
            image = Image.new("L", (eink_width, eink_height), "white")
            self._prepare_menu(image)
            # display animation and default status
            self.start_animation()
        else:
            pass # for now


    def _image_display(self, retrieve_key):
        self.page = 1
        # retrieve image
        action, num = retrieve_key
        file_prefix = f"./temp-{sd_baker.model_name}-{action}-{num}.png"
        if not os.path.exists(file_prefix):
            logging.info(f"file {file_prefix} not found")
            return
        image = Image.open(file_prefix)
        # prepare image and dialog
        image = self._prepare_menu(image)
        # display
        self.show_last_image(image)
        # wait for use click to continue



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
        # self.image_buffer.append(image)
        # self.image_preview_page = self.selection_idx[self.page] + 2
        # TODO update states


        # image finished cooking done
        self.cooking = False

    def show_last_image(self, image):
        # add some details
        # image = insert_image(self.image, self.image_buffer.pop(0))
        hex_pixels = image_to_header_file(image)
        # show and update states
        self.clear_screen()
        self.full_screen(hex_pixels)
        self.in_4g = True
        self.locked = False
    
        

# prepare assets
# hardcode model
model_list = ['/home/kevin/ai/models/4_anyloracleanlinearmix_v10-zelda-merge-onnx/_add_ons.json']
sd_baker = SdBaker()
# override
sd_baker.char_id = "perfect face, 1girl, wizard, wizard hat, cape, simple background,"
controller = Controller()

if __name__ == "__main__":
    controller.layout[0]() # page 0
    controller.load_model() # load model
    controller.pending_image = True

    try:
        while True:
            time.sleep(3)
            print(f"ping - \n")
            if controller.pending_image and not controller.cooking:
                # trigger sd cooking tasks
                logger.info("background tasks triggerd")
                controller.cooking = True
                # controller.trigger_background_job()

    except Exception:
        pass

    GPIO.cleanup()

