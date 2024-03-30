# from sshkeyboard import listen_keyboard
import os
import sys
import time 
import random
import json
import collections
from pathlib import Path
from einkDSP import einkDSP
from encoder import Encoder, Button, MultiButtonMonitor
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
        self.page = 0 # 0 : status display, 1 : image display
        self.pending_swap_prompt = None
        self.current_screen_selects = []
        self.layout = {
            0 : self._show_status,
            1 : self._image_display,
        }
        self.selection_idx = [0] * len(self.layout)
        self.display_cache = {
            0 : [text_to_image("Waifu doing ok    \(. > w < .)/ ")],
            1 : [text_to_image("[save]")] + [text_to_image("[back]")],
        }
        logging.info('Controller instance created')

        self.action_choices = {
            "feed" : ["eating cake", "eating apple", "drinking milk", "eating pizza"],
            "play" : ["happy face, play music", "happy face, ropes, role play", "reading book", "play tablet", "play phone"],
            "clean" : ["naked, nsfw, taking bath, bathing bubbles"],
        }
        # buffers
        self.image_buffer = []
        self.action_buffer = []

        # threading issues
        self.locked = False
        self.animation_thread = None
        self.stop_animation_event = threading.Event()
        self.background_thread = None

        self.cooking = False
        self.pending_image = False
        self.image_preview_page = None

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
                dialog_image = ui_assets["small_dialog_box"]["image"],
                box_mat = self.display_cache[self.page],
                highligh_index = self.selection_idx[self.page] if self.action_buffer != [] else None,
                placement_pos = ui_assets["small_dialog_box"]["placement_pos"]
            )
        elif self.page == 1:
            image_with_dialog = apply_dialog_box(
                input_image = image,
                dialog_image = ui_assets["small_dialog_box"]["image"],
                box_mat = self.display_cache[self.page],
                highligh_index = self.selection_idx[self.page],
                placement_pos = ui_assets["small_dialog_box"]["placement_pos"]
            )
        return image_with_dialog


    def trigger_background_job(self):
        if self.background_thread is None or not self.background_thread.is_alive():
            # pick a prmpt from action_choices
            action = random.choice(list(self.action_choices.keys()))
            prompt = random.choice(self.action_choices[action])
            prefix = f"temp-{sd_baker.model_name}-{len(self.action_buffer)}"
            logger.info("background_task triggered")
            self.cooking = action
            self.background_thread = threading.Thread(target=self.sd_process, args=(prompt, prefix))
            self.background_thread.start()
        

    def sd_process(self, prompts=None, prefix=""):
        event = sd_baker.generate_image(add_prompt=prompts, callback=self.sd_image_callback, prefix=prefix)
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

    def stop_background_thread(self):
        logging.info("stop background_thread ")
        if self.background_thread:
            self.background_thread.join()
            self.background_thread = None

    def show_animation(self):
        self.locked = True
        start_time = time.time()

        action = random.choice([x[0] for x in self.action_buffer]) if self.action_buffer != [] else "idle" 
        frame0 = animation_2frame(self.image, path= f"./tamagochiChar/{action}/animation_0.png")
        frame1 = animation_2frame(self.image, path= f"./tamagochiChar/{action}/animation_1.png")
        # frame0 = draw_text_on_dialog("COOKING...", frame0, (eink_width//2, eink_height//3*2), (eink_width//2+50, eink_height//3*2), True)
        # frame1 = draw_text_on_dialog("COOKING...", frame1, (eink_width//2, eink_height//3*2), (eink_width//2+50, eink_height//3*2), True)
        while not self.stop_animation_event.is_set():
            # draw_text_on_img("{:.0f}s".format(time.time() - start_time), frame0)            
            pixels = dump_2bit(np.array(frame0.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32)).tolist()
            self.part_screen(pixels)
            time.sleep(1.5)
            # draw_text_on_img("{:.0f}s".format(time.time() - start_time), frame1)
            pixels = dump_2bit(np.array(frame1.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32)).tolist()
            self.part_screen(pixels)
            time.sleep(1.5)

        self.locked = False


    def _show_status(self, key):
        self.page = 0
        self._status_check()
        # check status list
        status = collections.Counter([x[0] for x in self.action_buffer])
        logger.info(status)
        logger.info(self.action_buffer)

        if not self.action_buffer:  # idle set dialog display
            # update display cache
            self.display_cache[self.page] = [text_to_image("Waifu doing ok    \(. > w < .)/ ")]
            # prepare image and dialog
            image = Image.new("L", (eink_width, eink_height), "white")
            self.image = self._prepare_menu(image)
            # display animation and default status
        else:
            if key == "up":
                self.selection_idx[self.page] = (self.selection_idx[self.page] - 1) % len(self.display_cache[self.page])
            elif key == "down":
                self.selection_idx[self.page] = (self.selection_idx[self.page] + 1) % len(self.display_cache[self.page])
            self.display_cache[0] = [text_to_image(f"{k}[{v}]") for k, v in status.items()]
            # add menu
            image = Image.new("L", (eink_width, eink_height), "white")
            self.image = self._prepare_menu(image)
        # if self.animation_thread and self.animation_thread.is_alive() : self.stop_start_animation() # stop animation

        
        if key == "enter":
            logging.info(f"hit enter")
            # lock
            self.locked = True
            self.stop_start_animation()
            self.clear_screen() # prepare for 4g display
            # pick the selected action
            logger.info(status)
            logger.info(self.action_buffer)
            action_choice = list(status.keys())[self.selection_idx[self.page]]
            logging.info(f"action_choice {action_choice}")

            # reset selection
            self.selection_idx[self.page] = 0

            # get and pop the action from buffer
            for i, action in enumerate(self.action_buffer):
                if action[0] == action_choice:
                    action, id = self.action_buffer.pop(i)
                    file_prefix = f"./temp-{sd_baker.model_name}-{id}.png"
                    if not os.path.exists(file_prefix):
                        logging.info(f"file {file_prefix} not found")
                        return
                    image = Image.open(file_prefix)
                    # prepare image and dialog
                    # image = self._prepare_menu(image)
                    # display
                    logging.info(f"disply --- triggerd ---- ")
                    self._image_display("init")
                    self.show_last_image(image)
                    return

        # update screen
        # self.update_screen()     
        # if not self.animation_thread or not self.animation_thread.is_alive() : 
        self.stop_start_animation()
        self.start_animation() # refresh animation  

    def _image_display(self, key):
        self.page = 1        
        if key == "enter": # get back to status page
            self._show_status("init")
            return


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
        self.action_buffer.append((self.cooking, len(self.action_buffer)))
        self.stop_background_thread()
        self.cooking = None


    def show_last_image(self, image):
        self.locked = True
        post_img = process_image(image)
        image_with_dialog = self._prepare_menu(post_img)
        hex_pixels = image_to_header_file(image_with_dialog)
        # show and update states
        self.clear_screen()
        self.full_screen(hex_pixels)
        self.in_4g = True
        self.locked = False
    
        

controller = Controller()

# prepare assets
# hardcode model
model_list = ['/home/kevin/ai/models/4_anyloracleanlinearmix_v10-zelda-merge-onnx/_add_ons.json']
sd_baker = SdBaker()
# sd_baker.width = 128*4
# sd_baker.height = 128*6
# override
sd_baker.char_id = "best quality,masterpiece, perfect face, 1girl,solo, peer proportional face,  wizard, wizard hat, black cape, simple background,looking at viewer"
controller = Controller()

if __name__ == "__main__":
    controller.layout[0](0) # page 0
    controller.load_model() # load model
    previous_actions = controller.action_buffer
    try:
        while True:
            time.sleep(5)
            print(f"ping - \n")
            if not controller.cooking:
                logger.info("background tasks triggerd")
                if len(controller.action_buffer) < 5:
                    controller.trigger_background_job()

            # if len(controller.action_buffer) != previous_action_num
            # previous_action_num = len(controller.action_buffer)
            if not controller.locked and controller.page == 0 and controller.action_buffer != previous_action_num:
                controller.layout[0](0) # page 0 poke
                previous_action_num = controller.action_buffer

    except Exception:
        pass

    GPIO.cleanup()

