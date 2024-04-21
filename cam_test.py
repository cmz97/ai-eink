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


from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import cv2 
import numpy as np
from picamera2 import Picamera2
from optimum.onnxruntime import ORTStableDiffusionInpaintPipeline
import torch
import math

eink = einkDSP()

eink.epd_init_fast()
eink.PIC_display_Clear()

camera = Picamera2()
logging.info(' taking pic >>> ')

preview_config = camera.create_preview_configuration({"size": (128*2, 128*3)})
capture_config = camera.create_still_configuration(main={"size": (128*2, 128*3)})
camera.configure(preview_config)
# camera.set_controls({"ExposureTime" : 0, "AeEnable":True}) 

# setting 
# minExpTime, maxExpTime = 500, 32000000
# camera.still_configuration.buffer_count = 2
# camera.still_configuration.controls.FrameDurationLimits = (minExpTime, maxExpTime)
# camera.configure("still")
camera.controls.AeEnable = True
# camera.controls.AeMeteringMode = 0
# camera.controls.Saturation = 1.0
# camera.controls.Brightness = 0.2
# camera.controls.Contrast = 1.0
# camera.controls.AnalogueGain = 1.0
# camera.controls.Sharpness = 1.0

# camera.start_and_capture_file("./image.jpg", delay=0.5,show_preview=False,capture_mode='preview')
camera.start()
time.sleep(2)

timer = time.time()
while time.time() - timer < 5:
    time.sleep(0.7)
    init_image = camera.switch_mode_and_capture_image(capture_config)
    w,h = init_image.size
    # print(w,h)
    # crop_box = (w//2-128, h//2-192, w//2+128, h//2+192)
    # init_image = init_image.crop(crop_box) 
    init_image.save('./cam_test_image.png')
    image = insert_image(Image.new("L", (eink_width, eink_height), "white"), init_image)
    pixels = dump_2bit(np.array(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.float32)).tolist()
    eink.epd_init_part()
    eink.PIC_display(pixels)
    time.sleep(0.7)

camera.close()


# # load and resize   
# init_image = Image.open("./image.jpg")
# crop_box = (640//2-128, 480//2-192, 640//2+128, 480//2+192)
# init_image = init_image.crop(crop_box) 
# init_image.save('./image.jpg')
# # Define an inference source
source = './cam_test_image.png'
# # Create a FastSAM model
model = FastSAM('../FastSAM-x.pt')  # or FastSAM-x.pt
logging.info("running sam inference >>>> ")
# # Run inference on an image
everything_results = model(source, device='cpu', retina_masks=True, imgsz=512, conf=0.55, iou=0.7)
# # Prepare a Prompt Process object
prompt_process = FastSAMPrompt(source, everything_results, device='cpu')
ann = prompt_process.text_prompt(text='hair, black hair')
masks = ann[0].masks.data.numpy()
mask = Image.fromarray(masks[0,:,:])
mask = mask.filter(ImageFilter.MaxFilter(size=5)) # grow mask
mask.save('./mask.jpg')

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

