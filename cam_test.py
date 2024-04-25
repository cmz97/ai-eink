import os
import sys
import time 
import random
import json
from pathlib import Path
from einkDSP import einkDSP
from encoder import Encoder, Button
from PIL import Image, ImageFilter, ImageOps
from utils import * 
import threading  # Import threading module
import RPi.GPIO as GPIO
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# GPIO.cleanup()
from Drivers.SAM.sam import SAM


from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import cv2 
import numpy as np
from picamera2 import Picamera2
from optimum.onnxruntime import ORTStableDiffusionInpaintPipeline
import torch
import math

# eink = einkDSP()
# eink.epd_init_fast()
# eink.PIC_display_Clear()

# camera = Picamera2()
# logging.info(' taking pic >>> ')

# preview_config = camera.create_preview_configuration({"size": (128*2, 128*3)})
# capture_config = camera.create_still_configuration(main={"size": (128*2, 128*3)})
# camera.configure(preview_config)
# # camera.set_controls({"ExposureTime" : 0, "AeEnable":True}) 

# # setting 
# # minExpTime, maxExpTime = 500, 32000000
# # camera.still_configuration.buffer_count = 2
# # camera.still_configuration.controls.FrameDurationLimits = (minExpTime, maxExpTime)
# # camera.configure("still")
# camera.controls.AeEnable = True
# # camera.controls.AeMeteringMode = 0
# # camera.controls.Saturation = 1.0
# # camera.controls.Brightness = 0.2
# # camera.controls.Contrast = 1.0
# # camera.controls.AnalogueGain = 1.0
# # camera.controls.Sharpness = 1.0

# # camera.start_and_capture_file("./image.jpg", delay=0.5,show_preview=False,capture_mode='preview')
# camera.start()
# time.sleep(2)

# timer = time.time()
# while True:
#     time.sleep(0.25)
#     init_image = camera.switch_mode_and_capture_image(capture_config)
#     w,h = init_image.size
#     # print(w,h)
#     # crop_box = (w//2-128, h//2-192, w//2+128, h//2+192)
#     # init_image = init_image.crop(crop_box) 
#     init_image.save('./cam_test_image.png')
#     image = insert_image(Image.new("L", (eink_width, eink_height), "white"), init_image)
#     grayscale = image.transpose(Image.FLIP_LEFT_RIGHT).convert('L')
#     hex_pixels = dump_1bit_with_dithering(np.array(grayscale, dtype=np.float32))    
#     # pixels = dump_2bit(np.array(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32)).tolist()
#     eink.epd_init_part()
#     eink.PIC_display(hex_pixels)
#     time.sleep(0.25)

# camera.close()


# # load and resize   
# init_image = Image.open("./image.jpg")
# crop_box = (640//2-128, 480//2-192, 640//2+128, 480//2+192)
# init_image = init_image.crop(crop_box) 
# init_image.save('./image.jpg')
# # Define an inference source
# source = './cam_test_image.png'
# # # Create a FastSAM model
# model = FastSAM('/home/kevin/ai/models/FastSAM-x.pt')  # or FastSAM-x.pt
# logging.info("running sam inference >>>> ")
# # # Run inference on an image
# everything_results = model(source, device='cpu', retina_masks=True, imgsz=512, conf=0.55, iou=0.7)
# # # Prepare a Prompt Process object
# prompt_process = FastSAMPrompt(source, everything_results, device='cpu')
# ann = prompt_process.text_prompt(text='hair, black hair')
# masks = ann[0].masks.data.numpy()
# mask = Image.fromarray(masks[0,:,:])
# mask = mask.filter(ImageFilter.MaxFilter(size=5)) # grow mask
# mask.save('./mask.jpg')

# init_image = Image.open("./image.jpg")
# mask = Image.open("./mask.jpg")
# mask = mask.filter(ImageFilter.MaxFilter(size=5)) # grow mask
# # mask = dilated_mask.crop(crop_box)  
# width, height = init_image.size


# def merge_inpaint(init_image, image, mask):
#     init_image = init_image.convert("RGBA")
#     alpha_image = image.copy()
#     mask_image = mask.convert("L")
#     alpha_image.putalpha(mask_image)
#     combined_image = Image.alpha_composite(init_image, alpha_image)
#     return combined_image



# eink = einkDSP()
# image = insert_image(Image.new("L", (eink_width, eink_height), "white"), init_image)
# hex_pixels = image_to_header_file(image)
# eink.epd_w21_init_4g()
# eink.pic_display_4g(hex_pixels)
# eink.epd_sleep()

# pipe = ORTStableDiffusionInpaintPipeline.from_pretrained('./dreamshaper-8-inpainting-fused-onnx')
# logging.info("model loaded >>>> ")

# image = pipe(
#             'masterpiece,best quality, nsfw, see thru clothes, lingerie, sexy, adult,',
#             negative_prompt='safe for work, sfw, clothed, dressed,',
#             image=init_image,
#             mask_image=mask,
#             width=width,
#             height=height,
#             num_inference_steps=3,
#             guidance_scale=1.0,
#             # latents=ret[0]["masked_latent"].latent_dist.sample(),
#             # eta=1.0,
#         ).images[0]
# combined_image = merge_inpaint(init_image, image, mask)
# combined_image.save('enuked.png')


# image = insert_image(Image.new("L", (eink_width, eink_height), "white"), combined_image)
# hex_pixels = image_to_header_file(image)
# eink.epd_w21_init_4g()
# eink.pic_display_4g(hex_pixels)
# eink.epd_sleep()


def merge_inpaint(init_image, image, mask):
    init_image = init_image.convert("RGBA")
    alpha_image = image.copy()
    mask_image = mask.convert("L")
    alpha_image.putalpha(mask_image)
    combined_image = Image.alpha_composite(init_image, alpha_image)
    return combined_image

class Cam:
    def __init__(self, eink):
        self.eink = eink
        self.camera = Picamera2()
        # logging.info(' taking pic >>> ')
        self._config()
        self.preview_event = threading.Event()  # Event to signal file is ready

    def _config(self):
        preview_config = self.camera.create_preview_configuration({"size": (128*2, 128*3)})
        self.capture_config = self.camera.create_still_configuration(main={"size": (128*2, 128*3)})
        self.camera.configure(preview_config)
        self.camera.controls.AeEnable = True

    def preview(self):
        self.camera.start()
        time.sleep(2)
        while self.preview_event.is_set():
            time.sleep(0.25)
            init_image = self.camera.switch_mode_and_capture_image(self.capture_config)
            image = insert_image(Image.new("L", (eink_width, eink_height), "white"), init_image)
            grayscale = image.transpose(Image.FLIP_LEFT_RIGHT).convert('L')
            hex_pixels = dump_1bit_with_dithering(np.array(grayscale, dtype=np.float32))    
            self.eink.epd_init_part()
            self.eink.PIC_display(hex_pixels)
            time.sleep(0.25)
        self.camera.close()
        init_image.transpose(Image.ROTATE_180).save('./cam_test_image.png')

    def run(self):
        self.preview_event.set()  # Clear the event at the start
        threading.Thread(target=self.preview).start()
        

    
class Application:
    def __init__(self):
        self.eink = einkDSP()
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()
        
        self.image = Image.new("L", (eink_width, eink_height), "white")
        self.cam = Cam(self.eink)
        self.sam = SAM(self.press_callback)

        self.cam.run()

        # models
        self.fastsam = FastSAM('/home/kevin/ai/models/FastSAM-x.pt')
        self.pipe = ORTStableDiffusionInpaintPipeline.from_pretrained('/home/kevin/ai/models/dreamshaper-8-inpainting-fused-onnx')

    def eink_display_4g(self, hex_pixels):
        logging.info('eink_display_4g')
        self.eink.epd_w21_init_4g()
        self.eink.pic_display_4g(hex_pixels)
        self.eink.epd_sleep()
        self.in_4g = True

    def eink_init(self):
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()

    def eink_display_2g(self, hex_pixels):
        logging.info('eink_display_2g')
        if self.in_4g : 
            self.transit()
            self.in_4g = False

        self.eink.epd_init_part()
        self.eink.PIC_display(hex_pixels)

    def transit(self):
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()
    
    def clear_screen(self):
        # self.gui.clear_page()
        image = Image.new("L", (eink_width, eink_height), "white")
        hex_pixels = dump_1bit(np.array(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.uint8))
        self.eink.epd_init_part()
        self.eink.PIC_display(hex_pixels)

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
        
    def _fast_text_display(self, text="loading ..."):
        image = fast_text_display(self.image, text)
        grayscale = image.transpose(Image.FLIP_TOP_BOTTOM).convert('L')
        hex_pixels = dump_1bit_with_dithering(np.array(grayscale, dtype=np.float32))
        # hex_pixels = dump_1bit(np.array(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.uint8))
        self.part_screen(hex_pixels)


    def update_screen(self, image = None):
        if not image : image = self.image
        # image = self._prepare_menu(self.image)
        # update screen
        grayscale = image.transpose(Image.FLIP_TOP_BOTTOM).convert('L')
        logging.info('preprocess image done')
        hex_pixels = dump_1bit_with_dithering(np.array(grayscale, dtype=np.float32))
        logging.info('2bit pixels dump done')
        self.part_screen(hex_pixels)

    def run_sam(self):
        source = './cam_test_image.png'
        everything_results = self.fastsam(source,   device='cpu', retina_masks=True, imgsz=512, conf=0.55, iou=0.7)
        prompt_process = FastSAMPrompt(source, everything_results, device='cpu')
        ann = prompt_process.text_prompt(text='tops, shirt, clothes, topwear')
        masks = ann[0].masks.data.numpy()
        mask = Image.fromarray(masks[0,:,:])
        mask = mask.filter(ImageFilter.MaxFilter(size=5)) # grow mask
        mask.save('./mask.png')
        return mask
    
    def run_inpaint(self, init_image, mask):
        logging.info("running inpaint")
        # init_image = Image.open("./image.jpg")
        # mask = Image.open("./mask.jpg")
        # mask = mask.filter(ImageFilter.MaxFilter(size=5)) # grow mask
        width, height = init_image.size
        image = self.pipe(
                    'masterpiece, best quality, black dress, v-neck,',
                    negative_prompt='bad hand, bad face, worst quality, low quality, logo, text, watermark, username, harsh shadow, shadow, artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate',
                    image=init_image,
                    mask_image=mask,
                    width=width,
                    height=height,
                    num_inference_steps=3,
                    guidance_scale=1.0
                ).images[0]
        combined_image = merge_inpaint(init_image, image, mask)
        combined_image.save('enuked.png')
        return combined_image

    def press_callback(self, key):
        # take pic
        self.cam.preview_event.clear() # stop preview 
        time.sleep(2)
        source = Image.open("./cam_test_image.png")
        self.image = insert_image(Image.new("L", (eink_width, eink_height), "white"), source)
        # load image and process it
        self._fast_text_display("process [1] ...") # call for recording 
        # run sam
        mask = self.run_sam()
        # intermidiate display
        combined_image = merge_inpaint(source, Image.new('RGBA', (256, 384), "black"), mask)
        image = insert_image(Image.new("L", (eink_width, eink_height), "white"), combined_image)
        self.image = image
        hex_pixels = image_to_header_file(image)
        self.clear_screen()
        self.full_screen(hex_pixels)

        # send to inpaint
        self._fast_text_display("process [2] ...") # call for recording 
        combined_image = self.run_inpaint(source, mask)

        # display
        image = insert_image(Image.new("L", (eink_width, eink_height), "white"), combined_image)
        self.image = image
        hex_pixels = image_to_header_file(image)
        self.clear_screen()
        self.full_screen(hex_pixels)



if __name__ == "__main__":
    
    app = Application()
    
    while True:
        print("ping")
        time.sleep(0.5)