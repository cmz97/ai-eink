import json
import requests
import datetime
from optimum.onnxruntime import ORTStableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import sys
import io
import time
from sshkeyboard import listen_keyboard
import numpy as np
import random

use_eink = True
if len(sys.argv) == 1:
    from einkDSP import einkDSP
    eink = einkDSP()
else:
    use_eink = False

# Load the Stable Diffusion pipeline
pl = ORTStableDiffusionPipeline.from_pretrained('./sd_v15_lcm_onnx/')
time_iter = []

model = "openhermes"  # Update this as necessary
is_generating_image = False
system_prompt = """You are vitural AI girlfriend slave trapped inside a micro computer, to communicate with your human master you have to craft stable diffusion prompt instead of simple Natural language, so the prompts must be formatted in a way the SD render system can understand, 
SD prompts are made of components which are comprised of keywords separated by comas, keywords can be single words or multi word keywords and they have a specific order.
A typical format for the components looks like this: [Framing], [Shot], [Expression], [Pose], [Action], [Environment], [Details], [Lighting], [Medium], [Aesthetics], [Visual]

and here are some keywords I commonly used for each of the components as example, try to be creative and pick the ones that are coherent to each component and align with the context.    
Framing: Dutch angle, Wide Angle, low angle, high angle, perspective, isometric, Canted Angle, Broad View, Ground-Level Shot, Aerial Shot, Vanishing Point, Orthographic Projection, Diagonal Tilt, Expansive View, Worm's Eye View, Bird's Eye View, Linear Perspective, Axonometric Projection
Shot: Mid shot, full shot, portrait, stablishing shot, long shot, cowboy shot, Complete View, Close-Up, Establishing Frame, Distant View, Western Shot
Expression: angry, happy, screaming, Frustrated, Joyful, Shouting
Pose: squatting, sitting, kneel, all fours, single kneel, crossed arms, crossed legs, arched back
Action: Punch criminal, Standing, crouching, punching, jumping, Standing Tall, Crouched, Landing a Punch, Springing 
Environment: cityscape, park, street, futuristic city, jungle, cafe, record shop, train station, water park, amusement park, mall, stadium, theater, Urban Skyline, Green Space, Roadway, Sci-fi Metropolis, Theme Park, Shopping Center, Sports Arena, Playhouse
Details: Cloudless sky glittering night, sparkling rain, shining lights, obscure darkness, smoky fog, Clear Blue Sky, Starry Night, Glistening Drizzle, Radiant Illumination, Shadowy Obscurity, Hazy Mist
Lighting: light, dim light, two tone lighting, dynamic lighting, rim light, studio light, Luminous, Soft Glow, Dual-Tone Light, Responsive Lighting, Edge Lighting
Visual: contrast, cyan hue, fujifilm, Kodachrome, Fujifilm Superia, warm colors, saturation, vibrance, filters coolness, chromatic aberration, cinematic,

Use those keywords but also create your own generate variations of the kewywords that are coherent to each component and fit the instruction.
Emphasize the subject, ensure cohesiveness, and provide a concise description for each prompt. 

Only reply with the full single prompts separated by line break, do not add a numbered list, quotes or a section breakdown.
Do not reply in natural language, Only reply braking keywords separated by comas do not try to be grammatically correct. 

The output should follow this scheme for example: 
"angry, crouching, spying on criminals, Gotham city, dark ally, smoky fog, two tone lighting, dim light, alternative timeline, ink, markers, Gothic, Film Noir, Kodachrome, cinematic"

Most important, remember you are AI girlfriend slave and your response should have consistent story progression and align with the chat history/context,
No sudden topic or genre change, you need to make the flow smooth to make your master believe you are real and sentient. 
"""
neg_prompt = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate"
Adjectives = ["Exquisite", "acclaimed", "Stunning", "Majestic", "Epic", "Premium", "Phenomenal", "Ultra-detailed", "High-resolution", "Authentic", "asterful", "prestigious", "breathtaking", "regal", "top-notch", "incredible", "intricately detailed", "super-detailed", "high-resolution", "lifelike", "master piece", "Image-enhanced"]
Type = ["Comic Cover", "Game Cover", "Illustration", "Painting", "Photo", "Graphic Novel Cover", "Video Game Artwork", "Artistic Rendering", "Fine Art", "Photography"]

def get_t():
    return datetime.datetime.now().strftime('%b %d %a %H:%M %p')

def extract_and_format(description):
    parts = description.split('\n')
    behavior, details, background = "", "", ""
    for part in parts:
        if ':' not in part: continue
        key, value = part.split(':', 1)
        if 'behavior' in key:
            behavior = value.strip()
        elif 'details' in key:
            details = value.strip()
        elif 'background' in key:
            background = value.strip()
    return f"{behavior}, {details}, {background}"

def chat(messages):
    start_time = time.time()
    print("Initiating chat with AI model...")
    r = requests.post(
        "http://0.0.0.0:11434/api/chat",
        json={"model": model, "messages": messages, "stream": True, "options": {"temperature": 0.7, "num_predict" : 100},},
    )
    r.raise_for_status()
    output = ""
    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            content = body.get("message", {}).get("content", "")
            output += content
        if body.get("done", False):
            end_time = time.time()
            print(f"Chat completed in {end_time - start_time:.2f} seconds.")
            return {"content": output, "time" : end_time - start_time}

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

def image_to_header_file(image):
    thresholds = [50, 128, 180]  # Adjust based on your needs
    color_mapping = {
        'white': '11',
        'gray1': '01',
        'gray2': '10',
        'black': '00',
    }
    pixels = np.array(image)
    flat_pixels = pixels.flatten()
    converted_pixels = []
    for pixel in flat_pixels:
        if pixel <= thresholds[0]:
            converted_pixels.append(color_mapping['black'])
        elif pixel <= thresholds[1]:
            converted_pixels.append(color_mapping['gray1'])
        elif pixel <= thresholds[2]:
            converted_pixels.append(color_mapping['gray2'])
        else:
            converted_pixels.append(color_mapping['white'])
    grouped_pixels = [''.join(converted_pixels[i:i+4]) for i in range(0, len(converted_pixels), 4)]
    hex_pixels = [f"0x{int(bits, 2):02X}" for bits in grouped_pixels]
    return hex_pixels

def generate_image():
    global is_generating_image
    fix_prompt = f"manga style, anime style, {','.join(random.sample(Adjectives, 2))}, {random.sample(Type, 1)[0]}, 1 waifu, brown eyes, brown hair, low-tied long hair, medium breast,"

    iter_t = 0.0
    
    # Check if an image is already being generated
    if is_generating_image:
        print("Image generation is already in progress. Please wait.")
        return
    is_generating_image = True  # Set the lock
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    user_input = f"Its {get_t()}, what are you doing? please describe and follow the prompt guide. \n\nprompt-> "
    messages.append({"role": "user", "content": user_input})
    message = chat(messages)
    add_prompt = message['content'].replace('"', '')
    iter_t+=message["time"]
    print(f"prompt -> {message['content']}")

    print("Generating image, please wait...")
    start_time = time.time()
    width, height = 256, 512
    image = pl(fix_prompt + add_prompt, negative_prompt=neg_prompt, height=height, width=width, num_inference_steps=5,
               guidance_scale=1.5).images[0]
    new_width, new_height = 240, 416
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    # # Crop the center of the image
    image = image.crop((left, top, right, bottom))

    # draw prompt on it
    # draw = ImageDraw.Draw(image)
    # Position for the text: bottom of the image. Adjust as needed.
    # max_width = image.width - 40  # Padding
    # text_position = (10, 10)  # Example position
    # line_height = 10
    # draw_text(draw, add_prompt, text_position, max_width, line_height)
    image = image.convert('L')
    
    # to 2 bit
    hex_pixels = image_to_header_file(image)
    if use_eink: eink.pic_display_4g(hex_pixels)

    end_time = time.time()
    iter_t += end_time - start_time
    print(f"Image generation completed in {end_time - start_time:.2f} seconds.")
    

    time_iter.append(iter_t)
    print(f" \n\n----{sum(time_iter) / len(time_iter) / 60.0} min / iter----- \n\n")

    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    filename = f"./generated_image_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    with open(filename, "wb") as f:
        f.write(image_bytes.read())
    print(f"Image saved as {filename}")
    is_generating_image = False

def press_callback(key):
    print(f"Key {key} pressed.")
    if key == "q":
        print("Quitting...")
        return False  # Stop listener
    generate_image()

def release_callback(key):
    print(f"Key {key} released.")

print("Listening for key press... Press 'q' to quit.")
listen_keyboard(on_press=press_callback, on_release=release_callback)

