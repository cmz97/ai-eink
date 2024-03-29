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
from apps import SdBaker, PromptsBank, BookLoader
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

        # threading issues
        self.locked = False
        self.loading_screen_thread = None
        self.stop_loading_event = threading.Event()

        # model specific
        self.trigger_words = []

        # browsing temps
        self.browsing_file_list = [] 

    def _reload_prompt_cache(self):
        self.display_cache[3] = [
            [text_to_image(prompt)
            for prompt in prompts]
            for prompts in pb.prompt_selections
        ]

    def _update_prompt_cache(self, idx):
        self.display_cache[3][idx] = [text_to_image(prompts) for prompts in pb.prompt_selections[idx]]
        
    def start_loading_screen(self):
        self.locked = True
        logging.info("start_loading_screen thread ")
        self.stop_loading_event.clear()
        if self.loading_screen_thread is None or not self.loading_screen_thread.is_alive():
            self.loading_screen_thread = threading.Thread(target=self.show_loading_screen)
            self.loading_screen_thread.start()

    def stop_loading_screen(self):
        logging.info("stop_loading_screen thread ")
        self.stop_loading_event.set()
        if self.loading_screen_thread:
            self.loading_screen_thread.join()
            self.loading_screen_thread = None

    def show_loading_screen(self):
        self.locked = True
        start_time = time.time()
        frame0 = paste_loadingBox(self.image, frame=0)
        frame1 = paste_loadingBox(self.image, frame=1)
        frame0 = draw_text_on_dialog("GENERATING...", frame0, (eink_width//2, eink_height//3*2), (eink_width//2+50, eink_height//3*2), True)
        frame1 = draw_text_on_dialog("GENERATING...", frame1, (eink_width//2, eink_height//3*2), (eink_width//2+50, eink_height//3*2), True)
            
        while not self.stop_loading_event.is_set():
            time.sleep(0.5)
            draw_text_on_img("{:.0f}s".format(time.time() - start_time), frame0)
            
            pixels = dump_2bit(np.array(frame0.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32)).tolist()
            self.part_screen(pixels)
            time.sleep(0.5)
            draw_text_on_img("{:.0f}s".format(time.time() - start_time), frame1)
            pixels = dump_2bit(np.array(frame1.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32)).tolist()
            self.part_screen(pixels)
        self.locked = False

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

    def _status_check(self):
        if self.in_4g : 
            self.transit()
            self.in_4g = False
        
    def update_screen(self):
        image = self._prepare_menu(self.image)
        # update screen
        grayscale = image.transpose(Image.FLIP_TOP_BOTTOM).convert('L')
        pixels = np.array(grayscale, dtype=np.float32)
        logging.info('preprocess image done')
        hex_pixels = dump_2bit(pixels).tolist()
        logging.info('2bit pixels dump done')
        self.part_screen(hex_pixels)

    def load_model(self, model_path, model_info):
        self.locked = True
        self.transit()

        # just show one static image with loading , I'm sure load_model() blocks all threads
        image = Image.new("L", (eink_width, eink_height), "white")
        image = paste_loadingBox(image, frame=0)
        image = draw_text_on_dialog("LOADING MODEL ...", image, (eink_width//2-150, eink_height//2), (eink_width//2+150, eink_height//2), True)
        pixels = dump_2bit(np.array(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32)).tolist()
        self.part_screen(pixels)
        sd_baker.load_model(model_path, model_info['name'], model_info['trigger_words'])
        self.locked = False
    
    def _select_book(self, key):
        self.locked = True
        # sync status
        self.page = 0
        curr_file = str(model_list[self.selection_idx[self.page]])
        
        # process key
        if key == "up" : self.selection_idx[self.page] += 1 
        elif key == "down" : self.selection_idx[self.page] -= 1 
        elif key == "enter" :  # load book
            # load and proceed book
            ebk.loadFile(curr_file)
            self._read_page("init")
            return

        # rolling        
        self.selection_idx[self.page] = self.selection_idx[self.page] % len(model_list)

        # print screen
        thumbnail_image = render_thumbnail_page(Image.open("/".join(curr_file.split("/")[0:-1])+"/thumbnail.png"), "")        
        hex_pixels = image_to_header_file(thumbnail_image)
        self.full_screen(hex_pixels)
        self.locked = False


    def _read_page(self, key): # main page
        self.locked = True
        # sync status
        self.page = 1
        # process key
        if key == "up" : self.selection_idx[self.page] -= 1 
        elif key == "down" : self.selection_idx[self.page] += 1 
        
        # rolling        
        self.selection_idx[self.page] = max(1, self.selection_idx[self.page]) # % len(self.display_cache[self.page])

        # loading resources
        text_to_display = ebk.getPage(self.selection_idx[self.page])
        logger.info(text_to_display)
        # images 
        line_img = [text_to_image(line) for line in text_to_display]
        image = draw_text_on_screen(line_img)
        
        # print screen
        hex_pixels = image_to_header_file(image)
        self.full_screen(hex_pixels)
        self.locked = False

    def _display_page(self, key):
        if not self.model or self.locked: return

        # status sync
        self.page = 1

        # check key states
        if key == "init" : # show last
            # render
            logging.info(f"enter page {self.page}")
            file_cache = f"./temp-{self.model}.png"
            if os.path.exists(file_cache): # reload
                    image = Image.open(file_cache)
                    prompt_str = image.info.get("prompt")
                    pb.load_prompt(prompt_str)  
                    self._reload_prompt_cache()
                    self.sd_image_callback(image)
            else: # show empty, init dependencies
                # pb.fresh_prompt_selects()
                self._reload_prompt_cache()
                image = Image.new("L", (eink_width, eink_height), "white")
                self.image_callback(image)
            return 
        
        if key == "up" : self.selection_idx[self.page] -= 1
        elif key == "down" : self.selection_idx[self.page] += 1
        elif key == "enter" :  # hit prompt selection
            if self.selection_idx[self.page] == 0: 
                self._edit_prompt_page_0("init")
                return
            elif self.selection_idx[self.page] == 1:
                self._butSubmit()
                return 
            else:
                self._model_select_page("")
                return
            
        # render selection
        self._status_check()

        # rolling pin
        self.selection_idx[self.page] = self.selection_idx[self.page] % len(self.display_cache[self.page])
        # update screen
        self.update_screen()        

    def _edit_prompt_page_0(self, key): # display page + prompt select
        if not self.model or self.locked: return

        # status sync
        self.page = 2    
            
        if key == "up" : self.selection_idx[self.page] -= 1 
        elif key == "down" : self.selection_idx[self.page] += 1 
        elif key == "enter" :  # hit prompt selection
            if self.selection_idx[self.page] != len(self.display_cache[self.page])-1:
                # render selection
                self._status_check()
                # update and fresh new screen selects
                self._edit_prompt_page_1("init")
                return
            else:
                self._display_page("")
                return
                
        # render selection
        self._status_check()

        # rolling pin
        self.selection_idx[self.page] = self.selection_idx[self.page] % len(self.display_cache[self.page])
        # update screen
        self.update_screen()        

    def _edit_prompt_page_1(self, key): # prompt tuning
        # status sync
        self.page = 3
        if key == "init":
            self.update_screen()
            return
        if key == "up" : self.selection_idx[self.page] -= 1 
        elif key == "down" : self.selection_idx[self.page] += 1 
        elif key == "enter" :
            if self.selection_idx[self.page] == 0: 
                self._edit_prompt_page_0("")
                return # no change    
            # swap the prompt
            pb.update_prompt(self.selection_idx[self.page-1], self.selection_idx[self.page])
            # refresh cache
            self._update_prompt_cache(self.selection_idx[self.page-1])
            self.selection_idx[self.page] = 0 
            # back to last page
            self._edit_prompt_page_0("")
            return
        
        # rolling pin
        self.selection_idx[self.page] = self.selection_idx[self.page] % pb.selection_num # include self
        
        # update screen
        self.update_screen()

    def _butSubmit(self):
        # reset some status
        # self.selection_idx[self.page] = 0
        # run sd and render
        self.start_loading_screen()
        self.sd_process()
        pb.fresh_prompt_selects()
        self._reload_prompt_cache()


    def _image_browser(self, key):
        self.page = 4
        def load_cache():
            images_dir = './Asset/Fav'
            image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
            self.browsing_file_list = [f"{images_dir}/{f}" for f in image_files]
            self.browsing_file_list.sort() 
            return [text_to_image(f) for f in self.browsing_file_list] + [text_to_image("[back]")]

        if key == "init":
            # init dialog cache
            self.display_cache[self.page] = load_cache()
            # display first image
            image = Image.open(self.browsing_file_list[0])
            self.image_callback(image)
            return

        if key == "up" : self.selection_idx[self.page] -= 1 
        elif key == "down" : self.selection_idx[self.page] += 1 
        elif key == "enter" :  # load image
            if self.selection_idx[self.page] == len(self.display_cache[self.page])-1:
                # back to main page
                self._model_select_page("init")
                return
            image = Image.open(self.browsing_file_list[self.selection_idx[self.page]])
            self.image_callback(image)
            return

        # rolling        
        self.selection_idx[self.page] = self.selection_idx[self.page] % len(self.display_cache[self.page])
        
        # render selection
        self._status_check()
        # update screen
        self.update_screen()    

    def sd_process(self, prompts=None):
        if not prompts: prompts = pb.to_str() 
        # Start a new thread for the loading screen
        # prepare prompt and trigger gen
        event = sd_baker.generate_image(add_prompt=prompts, callback=self.sd_image_callback)
        event.wait()

    def press_callback(self, key):
        if self.locked : return
        print(f"Key {key} pressed.")
        if key == "q":
            print("Quitting...")
            exit()
        
        self.layout[self.page](key)
        
    def _update_text_box(self, text):
        # case needed : submit prompt tuning and when first time into play page or enter prompt tune page
        sub_page_id = self.selection_idx[self.page-1] # get the cache based on previous page selection
        self.display_cache[self.page][sub_page_id] = get_all_text_imgs(text)

    def _prepare_menu(self, image):     
        if self.page == 1 or self.page == 2:
            # def apply_dialog_box(input_image, dialog_image, box_mat, highligh_index, placement_pos):
            image_with_dialog = apply_dialog_box(
                input_image = image,
                dialog_image = ui_assets["small_dialog_box"]["image"],
                box_mat = self.display_cache[self.page],
                highligh_index = self.selection_idx[self.page],
                placement_pos = ui_assets["small_dialog_box"]["placement_pos"]
            )
        elif self.page == 3:
            sub_page_id = self.selection_idx[self.page-1] # get the cache based on previous page selection
            image_with_dialog = apply_dialog_box(
                input_image = image,
                dialog_image = ui_assets["large_dialog_box"]["image"],
                box_mat = self.display_cache[self.page][sub_page_id],
                highligh_index = self.selection_idx[self.page],
                placement_pos = ui_assets["large_dialog_box"]["placement_pos"]
            )
        elif self.page == 4:
            image_with_dialog = apply_dialog_box(
                input_image = image,
                dialog_image = ui_assets["large_dialog_box"]["image"],
                box_mat = self.display_cache[self.page],
                highligh_index = self.selection_idx[self.page],
                placement_pos = ui_assets["large_dialog_box"]["placement_pos"]
            )


        return image_with_dialog

    def sd_image_callback(self, image):
        post_img = process_image(image)
        image_with_dialog = self._prepare_menu(post_img)
        hex_pixels = image_to_header_file(image_with_dialog)
        # show and update states
        self.stop_loading_screen()
        self.clear_screen()
        time.sleep(0.3)
        self.full_screen(hex_pixels)
        self.in_4g = True
        self.image = post_img
        self.locked = False
    
    def image_callback(self, image):
        image_with_dialog = self._prepare_menu(image)
        hex_pixels = image_to_header_file(image_with_dialog)
        # show and update states
        self.stop_loading_screen()
        self.full_screen(hex_pixels)
        self.in_4g = True
        self.image = image
        self.locked = False
        

def release_callback(key):
    print(f"Key {key} released.")

# book folder
model_list = ['/home/kevin/ai/books/dune/Dune.txt']

# init 
pb = PromptsBank()
# sd_baker = SdBaker(pb)
screen_buffer = 10
ebk = BookLoader(filePath='../models/Dune.txt', 
screenWidth=eink_width - screen_buffer*2,
screenHeight=eink_height - screen_buffer*2,
fontWidth=char_width, fontHeight=char_height)

controller = Controller()
file_cache = "./temp.png"

if __name__ == "__main__":
    # sd_baker = SdBaker(pb)
    # start with main screen
    controller.layout[0]('init')
        
    try:
        while True:
            time.sleep(3)
            print(f"ping - \n")
    except Exception:
        pass

    GPIO.cleanup()

