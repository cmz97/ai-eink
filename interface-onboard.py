import os
import sys
import time 
import random
from einkDSP import einkDSP
from encoder import Encoder, Button
from PIL import Image
from utils import * 
import threading  # Import threading module


locked = False

# prepare file windows
images_dir = "./Asset"
buffer = (None, None)
init = False
file_idx = 0
eink = einkDSP()


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
    
def part_screen(hex_pixels):
    eink.epd_init_part()
    eink.PIC_display(hex_pixels)

def full_screen(hex_pixels):
    eink.epd_w21_init_4g()
    eink.pic_display_4g(hex_pixels)
    eink.epd_sleep()

def display_current_image_info(idx):
    global image_files, init
    # Acquire the lock
    text = get_file_list(idx)
    print(idx, text)
    # image = Image.open(f"{images_dir}/{image_files[idx]}")
    dialogBox = draw_text_on_dialog(text)
    updated_img = override_dialogBox(buffer[1], dialogBox)
    hex_pixels = dump_2bit(updated_img)
    if not init :
        transit()
        init = True
    part_screen(hex_pixels)

def rotChanged(counter, direction):
    global file_idx
    # page check
    # print("\033[H\033[J", end="")  # Clear the screen 
    print("* Direction: {}".format(direction))
    file_idx = counter % total_len
    display_current_image_info(file_idx)
    time.sleep(0.1)
            
def butClicked():
    global init, file_idx, buffer
    print(f"* butClicked for {file_idx}")  
    if buffer[0] == file_idx : return
    file_name = image_files[file_idx].replace("generated_image","dialogBox_image")
    image = Image.open(f"{images_dir}/{file_name}")
    # image = process_image(image)
    buffer = (file_idx, image)
    hex_pixels = image_to_header_file(image)
    full_screen(hex_pixels)
    init = False



def pauseDisplay():
    global locked, file_idx
    if not locked : print(f"* locking at {file_idx}")  
    else: print(f"* unlocking ... ")  
    locked = not locked
    
if __name__ == "__main__":

    if sys.argv[1] and sys.argv[1] == "display-mode":
        but = Button(26, callback=pauseDisplay) # gpio 26
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.png') and f.startswith('generated_image')]
        image_files.sort()  # Optional: Sort the files if needed
        total_len = len(image_files)
        display_pages = [x for x in range(0, total_len, 6)] # 6 per page

        while True :
            if locked : 
                time.sleep(5)
                continue           
        
            file_name = "dialogBox_image_seed_224137_20240304010040.png"#image_files[file_idx].replace("generated_image","dialogBox_image")
            print(f'displaying {file_name}')
            try:
                image = Image.open(f"{images_dir}/{file_name}")    
                hex_pixels = image_to_header_file(image)
                full_screen(hex_pixels)
            except Exception as e:
                print(f"{e}")
                #file_name = image_files[file_idx]
                #image = Image.open(f"{images_dir}/{file_name}")    
                #hex_pixels = image_to_header_file(image)
                #full_screen(hex_pixels)
            file_idx+=1
            #time.sleep(5)
            
        exit()

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png') and f.startswith('generated_image_seed_')]
    image_files.sort()  # Optional: Sort the files if needed
    total_len = len(image_files)
    display_pages = [x for x in range(0, total_len, 6)] # 6 per page

    rot = Encoder(22, 17, callback=rotChanged) # 22 17
    but = Button(26, callback=butClicked) # gpio 26
    eink = einkDSP()

    # first screen
    text = get_file_list(0)
    image = Image.open(f"{images_dir}/{image_files[0]}")
    dialogBox = draw_text_on_dialog(text)
    post_img = process_image(image, dialogBox)
    buffer = (0, post_img)
    hex_pixels = image_to_header_file(post_img)
    full_screen(hex_pixels)

    try:
        while True:
            time.sleep(3)
            print(f"ping - \n")
    except Exception:
        pass

    GPIO.cleanup()
