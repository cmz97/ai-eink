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
# from picamera2 import Picamera2

from optimum.onnxruntime import ORTStableDiffusionInpaintPipeline
import torch
import math


# camera = Picamera2()
# logging.info(' taking pic >>> ')

# preview_config = camera.create_preview_configuration({"size": (128*3, 128*2)})
# camera.configure(preview_config)
# camera.start_and_capture_file("./image.jpg",delay=0.5,show_preview=False,capture_mode='preview')


# # load and resize   
init_image = Image.open("./image.png")
w,h = init_image.size
crop_box = (w//2-128, h//2-192, w//2+128, h//2+192)
init_image = init_image.crop(crop_box) 
init_image.save('./image.png')
# Define an inference source
source = './image.png'
# Create a FastSAM model
model = FastSAM('../enuke/FastSAM-x.pt')  # or FastSAM-x.pt
logging.info("running sam inference >>>> ")
# Run inference on an image
everything_results = model(source, device='cpu', retina_masks=True, imgsz=512, conf=0.55, iou=0.7)
# Prepare a Prompt Process object
prompt_process = FastSAMPrompt(source, everything_results, device='cpu')
ann = prompt_process.text_prompt(text='human hair')
masks = ann[0].masks.data.numpy()
mask = Image.fromarray(masks[0,:,:])
mask.save('./mask.jpg')


# init_image = Image.open("./image.jpg")
# mask = Image.open("./mask.jpg")
mask = mask.filter(ImageFilter.MaxFilter(size=5)) # grow mask
# mask = dilated_mask.crop(crop_box)  
width, height = init_image.size


def merge_inpaint(init_image, image, mask):
    init_image = init_image.convert("RGBA")
    alpha_image = image.copy()
    mask_image = mask.convert("L")
    alpha_image.putalpha(mask_image)
    combined_image = Image.alpha_composite(init_image, alpha_image)
    return combined_image



eink = einkDSP()
image = insert_image(Image.new("L", (eink_width, eink_height), "white"), init_image)
hex_pixels = image_to_header_file(image)
eink.epd_w21_init_4g()
eink.pic_display_4g(hex_pixels)
eink.epd_sleep()

pipe = ORTStableDiffusionInpaintPipeline.from_pretrained('../models/dreamshaper-8-inpainting-fused-onnx')
logging.info("model loaded >>>> ")

image = pipe(
            'masterpiece, best quality, bald, shaved head',
            negative_prompt='(worst quality:2), (low quality:2), (normal quality:2)',
            image=init_image,
            mask_image=mask,
            width=width,
            height=height,
            num_inference_steps=3,
            guidance_scale=1.0,
            # latents=ret[0]["masked_latent"].latent_dist.sample(),
            # eta=1.0,
        ).images[0]
combined_image = merge_inpaint(init_image, image, mask)
combined_image.save('enuked.png')


image = insert_image(Image.new("L", (eink_width, eink_height), "white"), combined_image)
hex_pixels = image_to_header_file(image)
eink.epd_w21_init_4g()
eink.pic_display_4g(hex_pixels)
eink.epd_sleep()

