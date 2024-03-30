import json
import requests
import datetime
from optimum.onnxruntime import ORTStableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont, ImageOps
from PIL.PngImagePlugin import PngInfo
from diffusers import AutoencoderTiny

import sys
import re
import io
import time
import numpy as np
import random
import threading
import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PromptsBank:
    prompt_groups = ['body', 'clothing', 'face_1', 'face_2', 'look angle', 'pose', 'shot angle']
    with open('./prompt_pool.json') as f:
        prompts_bank = json.load(f)
    
    selection_num = 4

    def __init__(self):
        self.prompt_num = len(self.prompt_groups)
        self.prompt_selections = [random.sample(self.prompts_bank[group], self.selection_num) for group in self.prompt_groups]

    def fresh_prompt_selects(self):
        for idx, group in enumerate(self.prompt_groups):
            self.prompt_selections[idx][1:] = random.sample(self.prompts_bank[group], self.selection_num-1)        

    def load_prompt(self, prompt):
        self.fresh_prompt_selects()
        for idx, prompt_insert in enumerate(prompt.split(',')):
            self.prompt_selections[idx][0] = prompt_insert.strip()

    def to_str(self, delimiter=','):
        prompt = [x[0] for x in self.prompt_selections]
        return delimiter.join(prompt)

    def get_candidates_by_group(self, group):
        idx = self.prompt_groups.index(group)
        return self.prompt_selections[idx]

    def get_candidates_by_idx(self, idx):
        return self.prompt_selections[idx]

    def update_prompt(self, group_idx, idx):
        # swap 0 and idx
        self.prompt_selections[group_idx][0], self.prompt_selections[group_idx][idx] = self.prompt_selections[group_idx][idx], self.prompt_selections[group_idx][0]

    def get_prompts(self, group=None):
        if not group:
            ret = []
            for x in self.prompt_groups:
                ret.append(random.sample(self.prompts_bank[x], 1)[0])
            return ret
        else:
            return random.sample(self.prompts_bank[group], self.selection_num)  

class SdBaker:
    # CONSTANTS
    neg_prompt = "ng_deepnegative_v1_75t, bad hand, bad face, worst quality, low quality, logo, text, watermark, username, harsh shadow, shadow, artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate"
    char_id = "perfect face, 1girl, solo, peer proportional face, simple background, looking at viewer"
    # model_path =  '../anyloracleanlinearmix_v10-zelda-merge-onnx'
    width, height = 128*2, 128*3
    num_inference_steps = 3
    guidance_scale = 1.0

    def __init__(self, vae_override=None):
        self.pl = None
        # init func
        # self._load_model()
        logging.info('SdBaker instance created')

        # swap models
        self.model_path = None
        self.model_name = ""
        self.trigger_words = ""
        self.vae = None
        if vae_override:
            self.vae = AutoencoderTiny.from_pretrained(vae_override)

    def load_vae(self, vae_path):
        self.vae = AutoencoderTiny.from_pretrained(vae_path)

    def load_model(self, model_path, model_name, trigger_words):
        self.model_path = model_path
        self.model_name = model_name
        self.trigger_words = trigger_words

        # print(model_path, model_name, trigger_words)
        st = time.time()
        logging.info('start model loading')
        self.pl = None # release mem
        self.pl = ORTStableDiffusionPipeline.from_pretrained(self.model_path)
        logging.info(f'model loading done took {time.time() - st} sec')

    def _get_generator(self, seed = np.random.randint(0, 1000000) ):
        torch.manual_seed(seed)
        return np.random.RandomState(seed) , seed

    def generate_image(self, add_prompt="", callback=None):
        event = threading.Event()
        thread = threading.Thread(target=self._generate_image_thread, args=(add_prompt, event, callback))
        thread.start()
        return event

    def _generate_image_thread(self, add_prompt, event, callback=None, image_prefix=None):
        full_prompt = f"{self.char_id}, {self.trigger_words}, {add_prompt},"
        # logging.info(f" ingesting prompt : {full_prompt}")
        print("Generating image, please wait...")
        start_time = time.time()
        g, seed = self._get_generator()

        if self.vae:
            latents = self.pl(full_prompt,
                negative_prompt=self.neg_prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                generator=g,
                guidance_scale=self.guidance_scale,
                output_type="latent").images
            with torch.no_grad():
                latents = self.vae.decode(torch.from_numpy(latents) / self.vae.config.scaling_factor, return_dict=False)[0]
                do_denormalize = [True] * latents.shape[0]
                image = self.pl.image_processor.postprocess(latents.numpy(), output_type='pil', do_denormalize=do_denormalize)[0]
        else:
            image = self.pl(full_prompt,
                            negative_prompt=self.neg_prompt,
                            height=self.height,
                            width=self.width,
                            num_inference_steps=self.num_inference_steps,
                            generator=g,
                            guidance_scale=self.guidance_scale).images[0]
        print(f"Image generation completed in {time.time() - start_time:.2f} seconds.")
        
        # encode meta data and cache
        metadata = PngInfo()
        metadata.add_text("seed", str(seed))
        metadata.add_text("pre_prompt", self.char_id)
        metadata.add_text("prompt", add_prompt)
        metadata.add_text("neg_prompt", self.neg_prompt)
        metadata.add_text("height", str(self.height))
        metadata.add_text("width", str(self.width))
        metadata.add_text("num_inference_steps", str(self.num_inference_steps))
        metadata.add_text("guidance_scale", str(self.guidance_scale))

        self._save_img(image, metadata, image_prefix)

        if callback: callback(image)
        
        event.set()
    
    def _save_img(self, image, metadata, image_prefix):
        file_name = f"./temp-{self.model_name}.png" if not image_prefix else image_prefix+".png"
        image.save(file_name, pnginfo=metadata, optimize=True)


class BookLoader:
    def __init__(self, filePath, screenWidth, screenHeight, fontWidth, fontHeight):
        self.filePath = filePath
        self.charsPerLine = screenWidth // fontWidth
        self.linesPerPage = screenHeight // fontHeight
        self.currentPage = 1
        self.currentStartWordIndex = 0  # Track the start word index for the current page
        # self.loadFile(filePath)

    def loadFile(self, filePath):
        with open(filePath, 'r', encoding='ascii', errors='replace') as file:
            self.words = file.read().split()

    def getPage(self, pageNumber):
        if pageNumber < 1: return []  # Page number out of range
        
        if pageNumber < self.currentPage:
            # If going back, reset to start from the beginning
            self.currentStartWordIndex = 0
            self.currentPage = 1

        pageContent = []
        linesFilled = 0

        while self.currentStartWordIndex < len(self.words) and linesFilled < self.linesPerPage:
            lineContent, isNewLine = self._loadNextLine()
            # logging.info(lineContent)
            if isNewLine or self.currentStartWordIndex >= len(self.words):
                pageContent.append(lineContent)
                linesFilled += 1

            # Break if we've filled the page or reached the end of text, but only increment page counter if full page
            if linesFilled == self.linesPerPage or self.currentStartWordIndex >= len(self.words):
                if self.currentPage == pageNumber:
                    break
                else:
                    pageContent.clear()
                    linesFilled = 0
                    self.currentPage += 1

        return pageContent

    def _loadNextLine(self):
        line = []
        lineLength = 0
        isNewLine = False

        while self.currentStartWordIndex < len(self.words):
            word = self.words[self.currentStartWordIndex]
            # handle corner case
            if len(word) > self.charsPerLine:  # need to break word
                line.append(word[:self.charsPerLine])
                lineLength += self.charsPerLine + (1 if line else 0)
                self.currentStartWordIndex += 1
                continue

            # logging.info(f"word -> {word}")

            if lineLength + len(word) + (1 if line else 0) <= self.charsPerLine:
                # Add word to line
                line.append(word)
                lineLength += len(word) + (1 if line else 0)  # Add space for next word
                self.currentStartWordIndex += 1
            else:
                # Word doesn't fit, start a new line
                isNewLine = True
                break

        return " ".join(line), isNewLine

    def getProgress(self):
        # This is a simplistic progress measure, can be improved for accuracy
        return f"Page {self.currentPage}, approximately {self.currentStartWordIndex / len(self.words) * 100:.2f}% through the text"



class SceneBaker:
    def __init__(self, book_name):
        self.scene_buffer = [""]
        self.prompt = """<|user|>
Scene: {book_paragraph}
Convert the scene above into a visual picture.<|endoftext|>
<|assistant|>(response with 6 words total, separate by commas) :
"""
        self.book_name = book_name
    
    def get_next_scene(self, book_paragraph):
        def strip_all_numbers(s):
            pattern = r"\d+, "
            return re.sub(pattern, "", s)

        st = time.time()
        prompt = self.prompt.format(
            # book_name=self.book_name,
            # last_scene=self.scene_buffer[-1],
            book_paragraph=book_paragraph
        )
        data = {
            "prompt": prompt,
            "model": "stablelm2",
            # "format": "json",
            "stream": False,
            "options": {
                "temperature": 1.5, 
                "top_p": 0.99, 
                "top_k": 100,
                "num_ctx" : 500,
                "num_predict": 25,  
                # "stop": ["]"]
            },
        }
        # logging.info(f"llm trggerd, prompt -> {prompt} ")
        response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
        json_data = json.loads(response.text)["response"]
        text = strip_all_numbers(json_data.replace("\n", ",").replace('"', '').replace(".", ","))
        logging.info(f"llm return recived  ->  time took {time.time() - st}  sec")
        # update
        self.scene_buffer.append(text)
        return text





### legacy code below deal with it later
def llm_call(sd_prompt_mods):
    pick_idx = random.randint(0, len(sd_prompt_mods.split(",")) - 1)
    word2replace = sd_prompt_mods.split(",")[pick_idx].strip()
    type2replace = word_type.split(",")[pick_idx].strip() 
    prompt = f"""You are an expert on character design, given the following character description: 
{sd_prompt_mods.replace(word2replace, f"[{word2replace}]") }
generate 3 [{type2replace}] options that can be used to switch from [{word2replace}],
use one word for each option, Respond using JSON. Key names should with no backslashes, values should be one single phrase.
""" + """for example: 
{
  "Option 1": "phrase 1",
  "Option 2": "phrase 2",
  "Option 3": "phrase 3"
}"""
    print(prompt)
    data = {
        "keep_alive": "5m",
        "prompt": prompt,
        "model": "stablelm-zephyr",
        # "format": "json",
        "stream": False,
        "options": {
            "temperature": 1.5, 
            "top_p": 0.99, 
            "top_k": 100,
            # "stop": ["]"]
        },
        "max_tokens": 100,  
    }
    response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
    json_data = json.loads(response.text)["response"]
    return word2replace, json_data


def get_t():
    return datetime.datetime.now().strftime('%b %d %a %H:%M %p')

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
    """Apply Floyd-Steinberg dithering and convert image to a string array."""
    
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
    pixels_quantized = np.digitize(pixels, bins=[64, 128, 192], right=True)
    pixel_map = {0: '00', 1: '01', 2: '10', 3: '11'}
    pixels_string = np.vectorize(pixel_map.get)(pixels_quantized)
    converted_pixels = pixels_string.flatten().tolist()
    grouped_pixels = [''.join(converted_pixels[i:i+4]) for i in range(0, len(converted_pixels), 4)]
    int_pixels = [int(bits, 2) for bits in grouped_pixels]
    return np.array(int_pixels, dtype=np.uint8)

def padding(img, expected_size):
    
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def draw_text_on_dialog(dialog_image_path, ascii_table_image_path, text, text_area_start, text_area_end):
    # Load the dialog box image
    dialog_image = Image.open(dialog_image_path)

    # Load the ASCII character image asset
    ascii_table_image = Image.open(ascii_table_image_path)

    # Calculate the size of each character cell
    ascii_table_width, ascii_table_height = ascii_table_image.size
    char_width = ascii_table_width // 16
    char_height = ascii_table_height // 14

    # Calculate the position and size of the text area
    text_area_width = text_area_end[0] - text_area_start[0]
    text_area_height = text_area_end[1] - text_area_start[1]
    
    # Initialize the position for the first character
    x, y = text_area_start

    # Loop through each character in the text
    for char in text:
        # Calculate the ASCII value, then find the row and column in the ASCII image
        ascii_value = ord(char)
        if 32 <= ascii_value <= 255:
            row = (ascii_value - 32) // 16
            col = (ascii_value - 32) % 16
        else:
            continue  # Skip characters not in the range 32-255

        # Calculate the position to slice the character from the ASCII image
        char_x = col * char_width
        char_y = row * char_height

        # Slice the character image from the ASCII image
        char_image = ascii_table_image.crop((char_x, char_y, char_x + char_width, char_y + char_height))

        # Paste the character image onto the dialog box image
        dialog_image.paste(char_image, (x, y))

        # Move to the next character position
        x += char_width
        if x + char_width > text_area_end[0]:  # Newline if we run out of space
            x = text_area_start[0]
            y += char_height
            if y + char_height > text_area_end[1]:  # Stop if we run out of vertical space
                break

    return dialog_image


def process_image(image, dialogBox=None):
    eink_width, eink_height = 240, 416
    scale_factor = eink_width / width
    new_height = int(height * scale_factor)
    scaled_image = image.resize((eink_width, new_height), Image.ANTIALIAS)
    curImage = Image.new("L", (eink_width, eink_height), "white")
    # Paste the scaled image onto the white image, aligned at the top
    curImage.paste(scaled_image, (0, 0))
    if dialogBox:
        curImage.paste(dialogBox, (3, eink_height-dialogBox.height-4))
    return curImage

def generate_image(add_prompt=""):
    global is_generating_image
    fix_prompt = f"{','.join(random.sample(Adjectives, 2))}, {random.sample(Type, 1)[0]}, monochrome, {char_id}, nsfw,"
    iter_t = 0.0
    
    # Check if an image is already being generated
    if is_generating_image:
        print("Image generation is already in progress. Please wait.")
        return
    is_generating_image = True  # Set the lock
    
    full_prompt = fix_prompt + add_prompt
    print("Generating image, please wait...")
    start_time = time.time()
    
    seed = np.random.randint(0, 1000000)
    g = np.random.RandomState(seed)
    image = pl(full_prompt, negative_prompt=neg_prompt, height=height, width=width, num_inference_steps=3, generator=g, guidance_scale=1.0).images[0]
    dialogBox = draw_text_on_dialog(dialog_image_path, ascii_table_image_path, add_prompt + f"\n[{seed}]", text_area_start, text_area_end)
    curImage = process_image(image, dialogBox)
    hex_pixels = image_to_header_file(curImage)

    if use_eink: 
        eink.epd_w21_init_4g()
        eink.pic_display_4g(hex_pixels)
        # for i in range(100):
        #         newBox = draw_text_on_dialog(dialog_image_path, ascii_table_image_path, str(i), text_area_start, text_area_end)
        #         curImage.paste(newBox, (0, eink_height-dialogBox.height))
        #         hex_pixels = image_to_header_file(curImage)

        #         eink.epd_w21_init_4g()
        #         eink.pic_display_4g(hex_pixels)s
        eink.epd_sleep()


    end_time = time.time()
    iter_t += end_time - start_time
    print(f"Image generation completed in {end_time - start_time:.2f} seconds.")
    

    time_iter.append(iter_t)
    print(f" \n\n----{sum(time_iter) / len(time_iter) / 60.0} min / iter----- \n\n")

    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    filename = f"./Asset/generated_image_seed_{seed}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    with open(filename, "wb") as f:
        f.write(image_bytes.read())
    print(f"Image saved as {filename}")
    curImage.save(f"./Asset/dialogBox_image_seed_{seed}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    is_generating_image = False


#print("Listening for key press... Press 'q' to quit.")


