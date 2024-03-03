import json
import math
import requests
import datetime
from optimum.onnxruntime import ORTStableDiffusionPipeline
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
# import cv2
import torch
import time
from sshkeyboard import listen_keyboard
import numpy as np
from random import random

import torch

# Load the Stable Diffusion pipeline
pl = StableDiffusionPipeline.from_pretrained('Model/astranime_V6-lcm-lora-fused-classic')
# pl = StableDiffusionPipeline.from_single_file('Model/majicmixRealistic_v7.safetensors')
pl.disable_xformers_memory_efficient_attention()
pl.load_lora_weights("Lora/BeautyNwsjMajic2-01.safetensors", adapter_name="PerfectNwsjMajic")
pl.load_lora_weights('Lora/smile.safetensors', adapter_name="smile")
pl.load_lora_weights('Lora/ClothingAdjuster3.safetensors', adapter_name="cloth")
pl.load_textual_inversion('Model/ng_deepnegative_v1_75t.pt')
# pl.set_adapters(["PerfectNwsjMajic", "smile", "cloth"], adapter_weights=[0.7,0.0,0.8])
# pl.fuse_lora(adapter_names=["PerfectNwsjMajic", "smile", "cloth"])

time_iter = []

model = "stablelm-zephyr"  # Update this as necessary
is_generating_image = False
neg_prompt = "ng_deepnegative_v1_75t,harsh shadow, shadow, watermark, bad hand, bad face, artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate"
fix_prompt = f"PerfectNwsjMajic, 1 waifu, black hair, long hair, smile, cute, monochrome, anime style,greyscale, medium breast"

options = []

def generate_image():
    global is_generating_image, options
    # fix_prompt = f"anime style, pixelart, {','.join(random.sample(Adjectives, 2))}, {random.sample(Type, 1)[0]}, 1 waifu, brown eyes, brown hair, low-tied long hair, medium breast, nsfw,"

    iter_t = 0.0
    
    # Check if an image is already being generated
    if is_generating_image:
        print("Image generation is already in progress. Please wait.")
        return
    is_generating_image = True  # Set the lock


    print("Generating image, please wait...")
    print(f"Prompt: {fix_prompt}")
    start_time = time.time()
    width, height = 128*2, 128*3

    image = []
    randomSeed = int(random()*10000)
    print("seed: " + str(randomSeed))
    generator = torch.Generator(device="cpu").manual_seed(randomSeed)
    for i in np.arange(-0.2, 0.8, 0.2):
        pl.set_adapters(["PerfectNwsjMajic", "smile", "cloth"], adapter_weights=[0.7,i,i])
        pl.fuse_lora(adapter_names=["PerfectNwsjMajic", "smile", "cloth"])

        try:
            # Your image generation call here, assuming variables are defined
            curImage = pl(fix_prompt, negative_prompt=neg_prompt, height=height, width=width, num_inference_steps=5,
                    guidance_scale=1.0, generator=generator).images[0]
            # Store or process the generated image as needed
        except Exception as e:
            print(f"An error occurred during image generation: {e}")

        pl.unfuse_lora()

        image.append(curImage)
        image.append(floyd_steinberg_dithering(curImage))

        end_time = time.time()
        iter_t += end_time - start_time
        print(f"Image generation completed in {end_time - start_time:.2f} seconds.")

    combine_images_side_by_side(image)

    # image = pl(full_prompt, negative_prompt=neg_prompt, height=height, width=width, num_inference_steps=5,
    #             guidance_scale=1.0,generator=torch.manual_seed(0)).images[0]
    # end_time = time.time()
    # iter_t += end_time - start_time
    # print(f"Image generation completed in {end_time - start_time:.2f} seconds.")
    # image.show()

    time_iter.append(iter_t)
    print(f" \n\n----{sum(time_iter) / len(time_iter) / 60.0} min / iter----- \n\n")

    is_generating_image = False

def floyd_steinberg_dithering(image):
    """Apply Floyd-Steinberg dithering to convert image to 2-bit."""
    grayscale = image.convert('L')
    pixels = np.array(grayscale, dtype=np.float32)
    for y in range(pixels.shape[0]-1):
        for x in range(1, pixels.shape[1]-1):
            old_pixel = pixels[y, x]
            new_pixel = np.round(old_pixel / 85) * 85
            pixels[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            pixels[y, x+1] += quant_error * 7 / 16
            pixels[y+1, x-1] += quant_error * 3 / 16
            pixels[y+1, x] += quant_error * 5 / 16
            pixels[y+1, x+1] += quant_error * 1 / 16
    pixels = np.clip(pixels, 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))

def combine_images_side_by_side(images):
    # Calculate total width and maximum height
    total_width = sum(image.width for image in images)
    max_height = max(image.height for image in images)

    # Create a new image with the appropriate size
    combined_image = Image.new('RGB', (total_width, max_height))

    # Paste images side by side
    current_x = 0
    for image in images:
        combined_image.paste(image, (current_x, 0))
        current_x += image.width

    # Show the combined image
    combined_image.show()


generate_image()
