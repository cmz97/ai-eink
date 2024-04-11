import subprocess
import threading
import queue
import asyncio
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
# GPIO.cleanup()
from Ebook_GUI import EbookGUI

def format_text(line_buffer, word, boxWidth, boxHeight, fontWidth, fontHeight):
        # line buffer -> <line_idx, words>
        charsPerLine = boxWidth // fontWidth
        linesPerPage = boxHeight // fontHeight
        # assume buffer = list of words

        lineLength = len(" ".join(line_buffer[-1])) if line_buffer else 0
        # handle corner case
        if len(word) > charsPerLine:  # need to break word
            return line_buffer + word[:charsPerLine]
            
        if lineLength + len(word) + (1 if line_buffer else 0) <= charsPerLine: # append and update
            if line_buffer:
                line_buffer[-1].append(word)
            else:
                line_buffer = [[word]]
        else: # new line
            # check if new page
            if len(line_buffer) >= linesPerPage: # new page
                return [[word]]
            line_buffer.append([word])

        # update buffer and return
        return line_buffer 

class Controller:

    def __init__(self):
        self.eink = einkDSP()
        # self.butUp = Button(9, direction='up', callback=self.press_callback) # gpio 26
        # self.butDown = Button(22, direction='down', callback=self.press_callback) # gpio 26
        # self.butEnter = Button(17, direction='enter', callback=self.press_callback) # gpio 26
        self.in_4g = True
        self.image = gui.canvas
        
        # buffers
        self.prompt_buffer = ""
        self.text_buffer = []
        self.image_buffer = []

        # threading issues
        self.locked = False
        self.cooking = False
        self.pending_image = False
        self.image_preview_page = None

    def background_task(self):
        self.sd_process(self.prompt_buffer)

    def transit(self):
        self.locked = True
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()
        logging.info('transit to 2g done')
    
    def clear_screen(self):
        # self.eink.PIC_display_Clear()
        image = Image.new("L", (eink_width, eink_height), "white")
        pixels = dump_2bit(np.array(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32)).tolist()
        self.part_screen(pixels)
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
        image = self.image
        # image = self._prepare_menu(self.image)
        # update screen
        grayscale = image.transpose(Image.FLIP_TOP_BOTTOM).convert('L')
        pixels = np.array(grayscale, dtype=np.float32)
        logging.info('preprocess image done')
        hex_pixels = dump_2bit(pixels).tolist()
        logging.info('2bit pixels dump done')
        self.part_screen(hex_pixels)

    def load_model(self):
        logger.info("loading model")
        sd_baker.load_model(
            '/home/kevin/ai/models/sdxs-512-0.9-onnx',
            "sdxs",
            "")

    def trigger_background_job(self):
        self.cooking = True
        logger.info("background_task triggered")
        background_thread = threading.Thread(target=self.background_task, daemon=True)
        background_thread.start()
        # update flag
        self.pending_image = False
        logger.info("pending image flag off")

    def sd_process(self, prompts):
        sd_baker._generate_image_thread(prompts, self.sd_image_callback)
        
    def sd_image_callback(self, image):
        self.image_buffer.append(image)
        
        # image finished cooking done
        self.cooking = False

    async def show_last_image(self):
        if self.image_buffer:
            image = insert_image(self.image, self.image_buffer.pop(0))
            hex_pixels = image_to_header_file(image)
            self.clear_screen()
            self.full_screen(hex_pixels)
            self.in_4g = True
            self.locked = False
            await asyncio.sleep(10)  # Properly await sleeping

    
    def stream_text(self, text):
        self._status_check()
        # screen_buffer = 10
        w, h = gui.text_area[1][0] - gui.text_area[0][0] , gui.text_area[1][1] - gui.text_area[0][1]
        self.text_buffer = format_text(
            self.text_buffer, 
            text, 
            screenWidth= w,
            screenHeight= h,
            fontWidth=gui.font_size, 
            fontHeight=gui.font_size * 1.2)

        # call for screen update
        self.image = gui.draw_text_on_canvas(gui.canvas, self.text_buffer)
        self.update_screen()        

    def sd_check(self):
        if self.prompt_buffer.endswith('.') : # send to prompt
            if not self.cooking : self.trigger_background_job()
            self.prompt_buffer = "" # refresh
        else:
            pass 

def run_c_executable(model_file, temperature, max_token, prompt, output_queue):
    command = f"/home/kevin/ai/llama2.c/run {model_file} -t {temperature} -n {max_token} -i \"{prompt}\""
    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, universal_newlines=True, bufsize=1) as proc:
        while True:
            output = proc.stdout.read()  # Read the entire output as a single string
            if not output:  # If no more output, end of process
                break
            
            words = output.split()  # Split the output into words based on whitespace
            for word in words:
                output_queue.put(word)  # Put each word into the queue
        
        proc.stdout.close()
        return_code = proc.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)

async def async_output_generator(output_queue):
    while True:
        try:
            output = output_queue.get_nowait()  # Non-blocking get
            if output is None:  # Use None as a sentinel value to indicate completion
                break
            yield output
        except queue.Empty:
            await asyncio.sleep(0.01)  # Briefly yield control to allow other tasks to run


def run_c_executable_async(model_file, temperature, max_token, prompt, output_queue):
    thread = threading.Thread(target=run_c_executable, args=(model_file, temperature, max_token, prompt, output_queue))
    thread.start()
    return thread


# Example usage
model_file = "/home/kevin/ai/llama2.c/stories110M.bin"
temperature = 0.8
max_token = 256
prompt = "Once upon a time  "

async def main():
    output_queue = queue.Queue()
    c_thread = run_c_executable_async(model_file, temperature, max_token, prompt, output_queue)
    async for output in async_output_generator(output_queue):
        # print(output)  # Or handle the output as needed
        # pass in word to buffer
        await controller.show_last_image()        
        controller.stream_text(output)
        controller.prompt_buffer += " " + output
        controller.sd_check()
    c_thread.join()
    output_queue.put(None)  # Signal the generator that the process is done


# hardcoded parts
gui = EbookGUI()
controller = Controller()
sd_baker = SdBaker(vae_override="../models/sdxs-512-0.9/vae")
# override sd 
sd_baker.neg_prompt = "bad hand, bad face, worst quality, low quality, logo, text, watermark, username, harsh shadow, shadow, artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate"
sd_baker.char_id = "masterpiece,best quality, graphic novel, kid story illustration,"
sd_baker.num_inference_steps = 1
controller.load_model()

# Example usage
if __name__ == "__main__":
    asyncio.run(main())
    backCounter = 0
    try:
        while True:
            time.sleep(1)
            backCounter += 1 if GPIO.input(9) == 1 else 0
            if backCounter >= 5:
                os._exit(0)
    except Exception:
        # logger.errors(e)
        pass

