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

def display_menu():
    listen_keyboard(on_press=on_press)
    print("\033[H\033[J", end="")  # Clear the screen
    for i, option in enumerate(menu_options):
        if i == current_selection:
            print(f"> {option}")  # Highlight the current selection
        else:
            print(f"  {option}")

def on_press(key):
    global current_selection, loop_running
    if key == "up":
        current_selection = (current_selection - 1) % len(menu_options)
        display_menu()
    elif key == "down":
        current_selection = (current_selection + 1) % len(menu_options)
        display_menu()
    elif key == "enter":
        execute_current_selection()
    elif key == "q" and loop_running:
        # Mechanism to stop 'run_sd' and return to the main menu
        loop_running = False

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

def main():
    display_menu()
    
def run_sd():
    global loop_running
    # word_type = "accessories, clothes, facial details, facial expression," need to match with catagories
    sd_prompt_mods = "ear rings, black dress, Freckles, blushing"
    while loop_running:
        time.sleep(0.5)
        generate_image(sd_prompt_mods)
        if not loop_running:
            break
        try:
            st = time.time()
            rm_word, next_options = llm_call(sd_prompt_mods)
            print(f"{time.time() - st} sec to generate next prompt")
            options = list(next_options.values())
            picked_option = random.choice(options)
            sd_prompt_mods = sd_prompt_mods.replace(rm_word, picked_option.replace(",", "_"))
            print(f"new prompt: {sd_prompt_mods}")
        except Exception as e:
            print(f"Error calling LLM: {e} , \n {next_options}")
    
    display_menu() # back to main menu


def display_current_image_info(file_name):
    print(f"Current Image: {file_name}")

def on_press_gallery(key, image_files):
    global current_image_index
    locked = False
    if key == "up":
        current_image_index = (current_image_index - 1) % len(image_files)
        display_current_image_info(image_files[current_image_index])
    elif key == "down":
        current_image_index = (current_image_index + 1) % len(image_files)
        display_current_image_info(image_files[current_image_index])
    elif key == "enter" and not locked:
        locked = True
        # Display the selected image
        print(f"Selected Image: {image_files[current_image_index]}")
        image = Image.open(f"./Assets/{image_files[current_image_index]}")
        hex_pixels = image_to_header_file(image)
        eink.epd_w21_init_4g()
        eink.pic_display_4g(hex_pixels)
        eink.epd_sleep()
        locked = False
    elif key == "q":
        # Mechanism to stop gallery view and return to the main menu
        print("Exiting gallery...")
        display_menu()

def run_sd_gallery():
    images_dir = "./Assets"
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png') and f.startswith('dialogBox_image_seed_')]
    image_files.sort()  # Optional: Sort the files if needed
    display_current_image_info(image_files)
    listen_keyboard(on_press=on_press_gallery)    


if __name__ == "__main__":
    # Flag to control the execution of the loop
    loop_running = True
    current_image_index = 0
    main()
