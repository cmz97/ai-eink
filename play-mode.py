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
    image = Image.open(f"{images_dir}/{image_files[idx]}")
    dialogBox = draw_text_on_dialog(text)
    # updated_img = override_dialogBox(buffer[1], dialogBox)
    updated_img = process_image(image, dialogBox)
    # updated_img = override_dialogBox(image, dialogBox)

    grayscale = updated_img.transpose(Image.FLIP_TOP_BOTTOM).convert('L')
    pixels = np.array(grayscale, dtype=np.float32)

    print("hex_pixels" + str(time.time()))
    hex_pixels = dump_2bit(pixels).tolist()
    print("hex_pixels" + str(time.time()))

    if not init :
        print("transit" + str(time.time()))
        transit()
        print("transit" + str(time.time()))
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
    print(time.time())
    hex_pixels = image_to_header_file(image)
    print(time.time())
    full_screen(hex_pixels)
    print(time.time())
    init = False

def get_prompts(group=None):
    if not group:
        ret = []
        for x in prompt_groups:
            ret.append(random.sample(prompts_bank[x], 1)[0])
        return ",".join(ret)
    else:
        return random.sample(prompts_bank[group], 3) 


if __name__ == "__main__":
    rot = Encoder(22, 17, callback=rotChanged) # 22 17
    but = Button(26, callback=butClicked) # gpio 26
    eink = einkDSP()

    # first init
    text = get_prompts()

    # fix prompt 


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