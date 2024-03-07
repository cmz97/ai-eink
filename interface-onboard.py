import os
import time 
import random
from einkDSP import einkDSP
from encoder import Encoder, Button
from PIL import Image
from utils import * 
import threading  # Import threading module


lock = threading.Lock()

# prepare file windows
images_dir = "./Asset"
image_files = [f for f in os.listdir(images_dir) if f.endswith('.png') and f.startswith('generated_image_seed_')]
image_files.sort()  # Optional: Sort the files if needed
total_len = len(image_files)
file_idx = 0
display_pages = [x for x in range(0, total_len, 6)] # 6 per page
static_img = None
init = False

def get_file_list(idx):
    list_idx = range(max(0, idx-1), min(total_len, idx+6))
    text = ""
    for select_idx in list_idx:
        file_name = image_files[select_idx]
        seed = file_name.split('_')[-2]
        if select_idx == idx: # select
            text+=f"> {seed}\n"
        else:
            text+=f"{seed}\n"
    return text


def transit():
    eink.epd_init_fast()
    eink.PIC_display_Clear()
    
def fresh_screen(hex_pixels):
    eink.epd_init_part()
    eink.PIC_display(hex_pixels)
    
def display_current_image_info(idx):
    global image_files, init
    # Acquire the lock
    lock.acquire()
    try:
        # page check
        print("\033[H\033[J", end="")  # Clear the screen 
        text = get_file_list(idx)
        print(idx, text)
        # image = Image.open(f"{images_dir}/{image_files[idx]}")
        dialogBox = draw_text_on_dialog(text)
        updated_img = override_dialogBox(static_img, dialogBox)
        hex_pixels = dump_2bit(updated_img)
        if not init :
            transit()
            init = True
                    
        fresh_screen(hex_pixels)

            
    finally:
        # Always release the lock, even if an error occurred in the try block
        lock.release()


def rotChanged(counter, direction):
    print("* Direction: {}".format(direction))
    file_idx = counter % total_len
    display_current_image_info(file_idx)
    time.sleep(0.1)
            
def butClicked():
    print("* butClicked")  

rot = Encoder(22, 17, callback=rotChanged) # 22 17
but = Button(26, callback=butClicked) # gpio 26
eink = einkDSP()

if __name__ == "__main__":

    # first screen
    text = get_file_list(0)
    image = Image.open(f"{images_dir}/{image_files[0]}")
    dialogBox = draw_text_on_dialog(text)
    post_img = process_image(image, dialogBox)
    static_img = post_img
    hex_pixels = image_to_header_file(post_img)
    eink.epd_w21_init_4g()
    eink.pic_display_4g(hex_pixels)
    eink.epd_sleep()


    try:
        while True:
            time.sleep(3)
            print(f"ping - \n")
    except Exception:
        pass

    GPIO.cleanup()
