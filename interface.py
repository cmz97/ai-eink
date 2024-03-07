import time 
import random
import os
from sshkeyboard import listen_keyboard
from apps import generate_image, llm_call, image_to_header_file
from threading import Thread
from PIL import Image, ImageDraw, ImageFont, ImageOps
from einkDSP import einkDSP
eink = einkDSP()

menu_options = ["SD in loop", "SD Gallery", "Exit"]
current_selection = 0
page = 0 # 0 : main, 1 : sd, 2 : gallery 
image_files = [] 

def display_menu():
    global page
    page = 0 # reset

    print("\033[H\033[J", end="")  # Clear the screen
    for i, option in enumerate(menu_options):
        if i == current_selection:
            print(f"> {option}")  # Highlight the current selection
        else:
            print(f"  {option}")

def on_press(key):
    global current_selection, loop_running, current_image_index, image_files
    locked = False

    if key == "up":
        if page == 0: # main
            current_selection = (current_selection - 1) % len(menu_options)
            display_menu()
        elif page == 2 : 
            current_image_index = (current_image_index - 1) % len(image_files)
            display_current_image_info(current_image_index)
        else: 
            pass
    elif key == "down":
        if page == 0: # main
            current_selection = (current_selection + 1) % len(menu_options)
            display_menu()
        elif page == 2 : 
            current_image_index = (current_image_index + 1) % len(image_files)
            display_current_image_info(current_image_index)
        else:
            pass
    elif key == "enter":
        if page == 0: # main
            execute_current_selection()
        elif page == 2 and not locked: 
            locked = True
            # Display the selected image
            print(f"Selected Image: {image_files[current_image_index]}")
            image = Image.open(f"./Asset/{image_files[current_image_index]}")
            hex_pixels = image_to_header_file(image)
            eink.epd_w21_init_4g()
            eink.pic_display_4g(hex_pixels)
            eink.epd_sleep()
            locked = False            
    elif key == "q":
        loop_running = False
        display_menu()


def execute_current_selection():
    global loop_running
    if current_selection == len(menu_options) - 1:
        print("Exiting...")
        exit()  # Exit the program
    else:
        print(f"Executing {menu_options[current_selection]}")

        if current_selection == 0:
            # SD in loop
            loop_running = True
            Thread(target=run_sd).start()
        elif current_selection == 1:
            # SD Gallery
            loop_running = True
            Thread(target=run_sd_gallery).start()
        else:
            print("Invalid selection")
  
def run_sd():
    global loop_running, page
    page = 1 # set page

    # word_type = "accessories, clothes, facial details, facial expression," need to match with catagories
    sd_prompt_mods = "ear rings, black dress, Freckles, blushing"
    while loop_running:
        time.sleep(0.5)
        generate_image(sd_prompt_mods)
        if not loop_running:
            break
        
        st = time.time()
        rm_word, json_data = llm_call(sd_prompt_mods)
        print(f"{time.time() - st} sec to generate next prompt")

        try:
            start_index = json_data.find('{')
            end_index = json_data.rfind('}')
            json_str = json_data[start_index:end_index+1]
            next_options = json.loads(json_str)
            options = list(next_options.values())
            picked_option = random.choice(options)
            sd_prompt_mods = sd_prompt_mods.replace(rm_word, picked_option.replace(",", "_"))
            print(f"new prompt: {sd_prompt_mods}")
        except Exception as e:
            print(f"Error calling LLM: {e} , \n {json_data}")
    
    display_menu()

def display_current_image_info(idx):
    global image_files
    print("\033[H\033[J", end="")  # Clear the screen
    for i, file_name in enumerate(image_files):
        if i == idx:
            print(f"> {file_name}")  # Highlight the current selection
        else:
            print(f"  {file_name}")

def run_sd_gallery():
    global page, image_files
    page = 2 # switch page
    images_dir = "./Asset"
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png') and f.startswith('dialogBox_image_seed_')]
    image_files.sort()  # Optional: Sort the files if needed
    display_current_image_info(0)

if __name__ == "__main__":
    # Flag to control the execution of the loop
    loop_running = True
    current_image_index = 0
    display_menu()
    listen_keyboard(on_press=on_press)
