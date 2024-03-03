import json
import requests
import datetime
from optimum.onnxruntime import ORTStableDiffusionPipeline
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import sys
import io
import time
from sshkeyboard import listen_keyboard
import numpy as np
import random

# Load the Stable Diffusion pipeline
pl = StableDiffusionPipeline.from_pretrained('Model/astranime_V6-lcm-lora-fused-classic')
# pl = StableDiffusionPipeline.from_single_file('Model/majicmixRealistic_v7.safetensors')

pl.load_lora_weights("Lora/BeautyNwsjMajic2-01.safetensors", adapter_name="waifu")


pl.load_lora_weights('Lora/smile.safetensors', adapter_name="smile")

pl.load_lora_weights('Lora/ClothingAdjuster3.safetensors', adapter_name="cloth")

pl.set_adapters(["waifu", "smile", "cloth"], adapter_weights=[0.7,0.4,0.1])
pl.fuse_lora()

time_iter = []

model = "stablelm-zephyr"  # Update this as necessary
is_generating_image = False
neg_prompt = "harsh shadow, shadow, watermark, bad hand, bad face, artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate"
Adjectives = ["Exquisite", "acclaimed", "Stunning", "Majestic", "Epic", "Premium", "Phenomenal", "Ultra-detailed", "High-resolution", "Authentic", "asterful", "prestigious", "breathtaking", "regal", "top-notch", "incredible", "intricately detailed", "super-detailed", "high-resolution", "lifelike", "master piece", "Image-enhanced"]
Type = ["Comic Cover", "Game Cover", "Illustration", "Painting", "Photo", "Graphic Novel Cover", "Video Game Artwork", "Artistic Rendering", "Fine Art", "Photography"]

# prompt option design
word_type = "Framing, Expression, Pose"
sd_prompt_mods = "Dutch angle, smile, squatting"

options = []

def llm_call(sd_prompt_mods):
    pick_idx = random.randint(0, len(sd_prompt_mods.split(",")) - 1)
    word2replace = sd_prompt_mods.split(",")[pick_idx].strip()
    type2replace = word_type.split(",")[pick_idx].strip() 
    prompt = f"""SD prompts are made of components which are comprised of keywords separated by comas, keywords can be single words or multi word keywords and they have a specific order.
A typical format for the components looks like this: 
[Framing], [Shot], [Expression], [Pose], [Action], [Environment], [Details], [Lighting], [Medium], [Aesthetics], [Visual]

Given this SD prompt: 
{sd_prompt_mods}

Generate WordOption1 and WordOption2 thats different from the [{type2replace}] word "{word2replace}". 
Respond using JSON. Key names should with no backslashes, values should use plain ascii with no special characters
"""
    # print(prompt)
    data = {
        "prompt": prompt,
        "model": "stablelm-zephyr",
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.99, "top_k": 100},
        "max_tokens": 100,   
    }
    response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
    json_data = json.loads(response.text)

    ret = json.loads(json_data["response"])
    # print(ret)
    return word2replace, ret

def draw_text(draw, text, position, max_width, line_height):
    """
    Draw the text on the image with basic word wrapping.
    """
    lines = []  # List to hold lines of text
    words = text.split()
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        # Simple estimate: assume each character takes up roughly 6 pixels in width
        if len(test_line) * 6 < max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    lines.append(line)  # Add the last line

    x, y = position
    for line in lines:
        draw.text((x, y), line, fill=(0, 0, 0))
        y += line_height

def generate_image(curr_prompt):
    global is_generating_image, options
    # fix_prompt = f"anime style, pixelart, {','.join(random.sample(Adjectives, 2))}, {random.sample(Type, 1)[0]}, 1 waifu, brown eyes, brown hair, low-tied long hair, medium breast, nsfw,"
    fix_prompt = f"<lora:ClothingAdjuster3:0.3>, PerfectNwsjMajic, 1 waifu, glasses, , smile, cute, monochrome, anime style,greyscale, medium breast"

    iter_t = 0.0
    
    # Check if an image is already being generated
    if is_generating_image:
        print("Image generation is already in progress. Please wait.")
        return
    is_generating_image = True  # Set the lock

    # full_prompt = fix_prompt + curr_prompt
    full_prompt = fix_prompt

    print("Generating image, please wait...")
    print(f"Prompt: {full_prompt}")
    start_time = time.time()
    width, height = 128*2, 128*3
    image = pl(full_prompt, negative_prompt=neg_prompt, height=height, width=width, num_inference_steps=5,
               guidance_scale=1.0).images[0]
    end_time = time.time()
    iter_t += end_time - start_time
    print(f"Image generation completed in {end_time - start_time:.2f} seconds.")
    

    time_iter.append(iter_t)
    print(f" \n\n----{sum(time_iter) / len(time_iter) / 60.0} min / iter----- \n\n")

    # image_bytes = io.BytesIO()
    # image.save(image_bytes, format='PNG')
    # image_bytes.seek(0)
    # filename = f"./generated_image_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    # with open(filename, "wb") as f:
    #     f.write(image_bytes.read())
    # print(f"Image saved as {filename}")
    image.show()

    # prepare next options
    rm_word, next_options = llm_call(curr_prompt)
    options = [(rm_word, next_options['WordOption1']), (rm_word, next_options['WordOption2'])]
    print(f"Options: {options}")
    is_generating_image = False

def press_callback(key):
    global sd_prompt_mods
    print(f"Key {key} pressed.")
    if key == "q":
        print("Quitting...")
        exit()

    if options:
        if key == "1":
            print("Option 1 selected.")
            curr, next = options[0]
            sd_prompt_mods = sd_prompt_mods.replace(curr, next)
            generate_image(sd_prompt_mods)
        elif key == "2":
            print("Option 2 selected.")
            curr, next = options[1]
            sd_prompt_mods = sd_prompt_mods.replace(curr, next)
            generate_image(sd_prompt_mods)
        else:
            print("Invalid option. Please press 1 or 2.")
    else:
        print("Generating image...")
        generate_image(sd_prompt_mods)

def release_callback(key):
    print(f"Key {key} released.")

print("Listening for key press... Press 'q' to quit.")
listen_keyboard(on_press=press_callback, on_release=release_callback)

