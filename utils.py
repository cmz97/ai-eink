import numpy as np
from typing import Any, Dict, Optional, Union
from PIL import Image, ImageOps
from numba import jit
import bisect

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

dialog_image_path = 'dialogBox.png'
ascii_table_image_path = 'asciiTable.png'
ui_elements_path = 'ui_sheet.png'
text_area_start = (9, 12)
text_area_end = (226, 80)


# load image in ram to save time
dialog_image = Image.open(dialog_image_path)
ascii_table_image = Image.open(ascii_table_image_path)
ui_elements_image = Image.open(ui_elements_path)
loading_box_image1 = Image.open('./loading1_v1.png')
loading_box_image2 = Image.open('./loading2_v1.png')



ui_assets = {
    "small_dialog_box" : {
        "image" : ui_elements_image.crop((3, 263, 236, 324)),
        "placement_pos" : (3, 350)
    },
    "large_dialog_box" : {
        "image" : ui_elements_image.crop((3, 325, 236, 415)),
        "placement_pos" : (3, 325)
    },
    "reroll_button" : {
        "image" : ui_elements_image.crop((25, 32, 48, 55)),
        "position" : (25, 32, 48, 55)
    }
}

# Calculate the size of each character cell
ascii_table_width, ascii_table_height = ascii_table_image.size
char_width = ascii_table_width // 16
char_height = ascii_table_height // 14
eink_width, eink_height = 240, 416

ORT_TO_NP_TYPE = {
    "tensor(bool)": np.bool_,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int16)": np.int16,
    "tensor(uint16)": np.uint16,
    "tensor(int32)": np.int32,
    "tensor(uint32)": np.uint32,
    "tensor(int64)": np.int64,
    "tensor(uint64)": np.uint64,
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
}

import logging
logger = logging.getLogger(__name__)

class ORTModelTiledVaeWrapper(object):
    def __init__(
        self,
        wrapped,
        decoder: bool,
        window: int,
        overlap: float,
    ):
        import torch
        from optimum.onnxruntime.modeling_diffusion import ORTModelVaeDecoder, ORTModelVaeEncoder
        from diffusers import AutoencoderKL
        from diffusers.models.autoencoders.vae import DecoderOutput
        from diffusers.models.modeling_outputs import AutoencoderKLOutput
        self.wrapped = wrapped
        self.decoder = decoder
        self.tiled = False
        self.set_window_size(window, overlap)

    def set_tiled(self, tiled: bool = True):
        self.tiled = tiled

    def set_window_size(self, window: int, overlap: float):
        self.tile_latent_min_size = window
        self.tile_sample_min_size = window * 8
        self.tile_overlap_factor = overlap

    def __call__(self, latent_sample=None, sample=None, **kwargs):
        # convert latent/sample type to match model for mixed fp16/fp32 support
        # sample_dtype = next(
        #     (
        #         input.type
        #         for input in self.wrapped.session.get_inputs()
        #         if input.name == "sample" or input.name == "latent_sample"
        #     ),
        #     "tensor(float)",
        # )
        # sample_dtype = ORT_TO_NP_TYPE[sample_dtype]

        # if latent_sample is not None and latent_sample.dtype != sample_dtype:
        #     logger.debug("converting VAE latent sample dtype to %s", sample_dtype)
        #     latent_sample = latent_sample.astype(sample_dtype)

        # if sample is not None and sample.dtype != sample_dtype:
        #     logger.debug("converting VAE sample dtype to %s", sample_dtype)
        #     sample = sample.astype(sample_dtype)

        if self.tiled:
            if self.decoder:
                return self.tiled_decode(latent_sample, **kwargs)
            else:
                return self.tiled_encode(sample, **kwargs)
        else:
            if self.decoder:
                return self.wrapped(latent_sample=latent_sample)
            else:
                return self.wrapped(sample=sample)

    def __getattr__(self, attr):
        return getattr(self.wrapped, attr)

    def blend_v(self, a, b, blend_extent):
        for y in range(min(a.shape[2], b.shape[2], blend_extent)):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[
                :, :, y, :
            ] * (y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent):
        for x in range(min(a.shape[3], b.shape[3], blend_extent)):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
                :, :, :, x
            ] * (x / blend_extent)
        return b

    def tiled_encode(
        self, x, return_dict: bool = True
    ):
        r"""Encode a batch of images using a tiled encoder.
        Args:
        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is:
        different from non-tiled encoding due to each tile using a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        look of the output, but they should be much less noticeable.
            x (`torch.FloatTensor`): Input batch of images. return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`AutoencoderKLOutput`] instead of a plain tuple.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        with torch.no_grad():
            overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
            blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
            row_limit = self.tile_latent_min_size - blend_extent

            # Split the image into 512x512 tiles and encode them separately.
            rows = []
            for i in range(0, x.shape[2], overlap_size):
                row = []
                for j in range(0, x.shape[3], overlap_size):
                    tile = x[
                        :,
                        :,
                        i : i + self.tile_sample_min_size,
                        j : j + self.tile_sample_min_size,
                    ]
                    tile = torch.from_numpy(self.wrapped(sample=tile.numpy())[0])
                    row.append(tile)
                rows.append(row)
            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile
                    # to the current tile and add the current tile to the result row
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_extent)
                    result_row.append(tile[:, :, :row_limit, :row_limit])
                result_rows.append(torch.cat(result_row, dim=3))

            moments = torch.cat(result_rows, dim=2).numpy()
            if not return_dict:
                return (moments,)

        return AutoencoderKLOutput(latent_dist=moments)

    def tiled_decode(
        self, z, return_dict: bool = True
    ):
        r"""Decode a batch of images using a tiled decoder.
        Args:
        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled decoding is:
        different from non-tiled decoding due to each tile using a different decoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        look of the output, but they should be much less noticeable.
            z (`torch.FloatTensor`): Input batch of latent vectors. return_dict (`bool`, *optional*, defaults to
            `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)

        with torch.no_grad():
            overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
            blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
            row_limit = self.tile_sample_min_size - blend_extent

            # Split z into overlapping 64x64 tiles and decode them separately.
            # The tiles have an overlap to avoid seams between tiles.
            rows = []
            for i in range(0, z.shape[2], overlap_size):
                row = []
                for j in range(0, z.shape[3], overlap_size):
                    tile = z[
                        :,
                        :,
                        i : i + self.tile_latent_min_size,
                        j : j + self.tile_latent_min_size,
                    ]
                    decoded = torch.from_numpy(self.wrapped(latent_sample=tile.numpy())[0])
                    row.append(decoded)
                rows.append(row)

            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile
                    # to the current tile and add the current tile to the result row
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_extent)
                    result_row.append(tile[:, :, :row_limit, :row_limit])
                result_rows.append(torch.cat(result_row, dim=3))

            dec = torch.cat(result_rows, dim=2)
            dec = dec.numpy()

            if not return_dict:
                return (dec,)

        return DecoderOutput(sample=dec)


def draw_text_on_img(text, image):
    x, y = (240-150)//2 + 10 , (416-150)//2 + 10
    for idx, char in enumerate(text):
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
        image.paste(char_image, (x, y))
        # Move to the next character position
        x += char_width
        if x + char_width > text_area_end[0]:  # Newline if we run out of space
            x = text_area_start[0]
            y += char_height
            if y + char_height > text_area_end[1]:  # Stop if we run out of vertical space
                break
    return image


def get_dialog_text(box_mat, box_size, highligh_index):
    # TODO The text line should not be bigger than the bounding box
    def line_wrap(char_imgs, font_width, max_width):
        lines = []
        line = []
        width = 0
        for img in char_imgs:
            if width + font_width <= max_width:
                line.append(img)
                width += font_width
            else:
                lines.append(line)
                line = [img]
                width = font_width
        lines.append(line)
        return lines
        
    text_area_width, text_area_height = box_size 
    margin = 2
    text_start = (0+margin, 0+margin)
    text_end = (text_area_width-margin, text_area_height-margin)

    page = Image.new("L", (text_area_width, text_area_height * 5), "white") # cap at 5 boxes long
    # cursor
    x, y = text_start

    highlight_y = None

    # iter per print line
    for idx, line in enumerate(box_mat):
        # TODO assime all single box per row for now
        # check if line wrap
        rows = line_wrap(line, char_width, text_area_width)
        for row in rows:
            for char in row:
                if idx == highligh_index:
                    char = ImageOps.invert(char.convert('RGB'))
                    highlight_y = y
                page.paste(char, (x, y))
                x += char_width
            # update y
            if y + char_height > page.height:  # Stop if we run out of vertical space
                break
            x = text_start[0] # reset x
            y += char_height # next line
    
    # crop around highlight
    crop_top = max(0, highlight_y - box_size[1]//2)
    crop_bot = crop_top + box_size[1]
    crop_box = (0, crop_top, text_area_width, crop_bot)
    page = page.crop(crop_box)
    return page

def apply_dialog_box(input_image, dialog_image, box_mat, highligh_index, placement_pos):
    input_image_ref = input_image.copy()
    dialog_image_ref = dialog_image.copy()
    margin = 10
    box_size = dialog_image.size[0] - 2*margin, dialog_image.size[1] - 2*margin
    text_image = get_dialog_text(box_mat, box_size, highligh_index)
    dialog_image_ref.paste(text_image, (margin, margin))
    input_image_ref.paste(dialog_image_ref, (placement_pos[0], placement_pos[1]))
    return input_image_ref

def text_to_image(text):
    buffer = []
    for char in text:
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
        # Move to the next character position
        buffer.append(char_image)
    return buffer

    

# now using window sliding to fit
def get_all_text_imgs(text):
    x, _ = text_area_start
    buffer = []
    lines = text.split('\n')
    
    new_line_maps = []

    for line_idx, line in enumerate(lines):
        # init
        buffer.append([])
        line_bunddle = [len(buffer)]

        # iter 
        for char in line:
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
             # Move to the next character position
            x += char_width
            if x + char_width > text_area_end[0]:  # Newline if we run out of space
                x = text_area_start[0]
                buffer.append([])
                line_bunddle.append(len(buffer))
            buffer[-1].append(char_image)
        
        # update
        new_line_maps.append(line_bunddle)

    return buffer, new_line_maps

def augment_text_imgs(buffer, highlight_index_list):
    # static constants
    text_area_height = text_area_end[1] - text_area_start[1]
    window_size = text_area_height // char_height


    # get screen candidates, find our page loc
    highlight_index = highlight_index_list[0]
    page_turns = [x for x in range(0, len(buffer), window_size)]
    page_number = bisect.bisect_right(page_turns, highlight_index)

    # prepare window
    for line_idx, line in enumerate(buffer[page_number : page_number + window_size]):
        for char_idx, char in enumerate(line):
            if char == "\n" : continue
            if line_idx + page_number == highlight_index:
                buffer[line_idx + page_number][char_idx] = invert_image(char)
            

def draw_text_on_dialog(text, image_ref=None, text_area_start=text_area_start, text_area_end=text_area_end, aligned=False, highlighted_lines=[]):
    dialog_image_ref = dialog_image.copy() if not image_ref else image_ref

    # Calculate the position and size of the text area
    text_area_width = text_area_end[0] - text_area_start[0]
    text_area_height = text_area_end[1] - text_area_start[1]
    
    # Initialize the position for the first character
    x, y = text_area_start

    buffer = []
    lines = text.split('\n')
    for line_idx, line in enumerate(lines):
        for char in line:

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
            if not aligned : 
                if highlighted_lines and line_idx in highlighted_lines:
                    dialog_image_ref.paste(invert_image(char_image), (x, y))
                dialog_image_ref.paste(char_image, (x, y))
            else : buffer.append(char_image)

            # Move to the next character position
            x += char_width
            if x + char_width > text_area_end[0]:  # Newline if we run out of space
                x = text_area_start[0]
                y += char_height
                if y + char_height > text_area_end[1]:  # Stop if we run out of vertical space
                    break
        # new line
        x = text_area_start[0]
        y += char_height
        if y + char_height > text_area_end[1]:  # Stop if we run out of vertical space
            break
    
    # for mid aligned text
    if buffer:
        # calculate for x
        x_len = len(buffer) * char_width
        x = eink_width//2 - x_len // 2
        for char in buffer:
            dialog_image_ref.paste(char, (x, text_area_start[1]))
            x+=char_width

    return dialog_image_ref

def draw_text_on_screen(lines, buffer=10):
    # new canvas
    page = Image.new("L", (eink_width, eink_height), "white")

    # Calculate the position and size of the text area
    text_area_width = eink_width - buffer*2
    text_area_height = eink_height - buffer*2
    
    # Initialize the position for the first character
    text_area_start = (buffer, buffer)
    text_area_end = (eink_width - buffer, eink_height - buffer)
    x, y = text_area_start
    
    for line in lines:
        for char_image in line: 
            page.paste(char_image, (x, y))
            x += char_width # step next
            if x + char_width > text_area_end[0]:  # Newline if we run out of space
                x = text_area_start[0]
        y += char_height # new line
    return page

def render_thumbnail_page(thumbnail, text, border=False):
    button_size = (32,32)
    thumbnail_width, thumbnail_height = thumbnail.size
    if thumbnail_width >= eink_width or thumbnail_height >= eink_height:
        buffer = 50
        max_width = eink_width - buffer * 2
        max_height = eink_height - buffer * 2
        # Calculate the scaling factor
        width_ratio = max_width / thumbnail.width
        height_ratio = max_height / thumbnail.height
        scale_factor = min(width_ratio, height_ratio)
        # Calculate new dimensions
        new_width = int(thumbnail.width * scale_factor)
        new_height = int(thumbnail.height * scale_factor)
        thumbnail = thumbnail.resize((new_width,new_height), Image.ANTIALIAS)

    if border :
        thumbnail = ImageOps.expand(thumbnail, border=10, fill='white')
        thumbnail = ImageOps.expand(thumbnail, border=2, fill='black')

    thumbnail_width, thumbnail_height = thumbnail.size
    image = Image.new("L", (eink_width, eink_height), "white")
    image.paste(thumbnail, ((eink_width - thumbnail_width)//2, (eink_height - thumbnail_height)//2))
    up = ui_elements_image.crop((button_size[0], 0, button_size[0]*2, button_size[1]))
    down = ui_elements_image.crop((0,0, button_size[0], button_size[1]))
    image.paste(up, (eink_width//2 - button_size[0]//2, (eink_height - thumbnail_height) // 4 - button_size[0]//2), mask = up)
    image.paste(down, (eink_width//2 - button_size[0]//2, eink_height - (eink_height - thumbnail_height) // 4 - button_size[1]//2), mask = down)
    
    # titles
    if text : image = draw_text_on_dialog(text, image, (eink_width//2 - 75, eink_height//3 * 2 + 10), (eink_width//2 + 75, eink_height//3 * 2 + 10), True)
    return image


def process_image(image, dialogBox=None, height=128*3, width=128*2):
    scale_factor = eink_width / width
    new_height = int(height * scale_factor)
    scaled_image = image.resize((eink_width, new_height), Image.ANTIALIAS)
    curImage = Image.new("L", (eink_width, eink_height), "white")
    # Paste the scaled image onto the white image, aligned at the top
    curImage.paste(scaled_image, (0, 0))
    if dialogBox:
        curImage.paste(dialogBox, (3, eink_height-dialogBox.height-4))
    return curImage


def override_dialogBox(image, dialogBox):
    image.paste(dialogBox, (3, 416-dialogBox.height-4))
    logging.info('Dialog box overriden')
    return image


# def get_loading_screen(image):
#     frame1 = paste_loadingBox(image, loading_box_image1)
#     frame2 = paste_loadingBox(image, loading_box_image2)
#     return dump_2bit(np.array(frame1, dtype=np.float32)).tolist(), dump_2bit(np.array(frame2, dtype=np.float32)).tolist()
        

def paste_loadingBox(image, frame):
    # loading_box size = 150 x 150
    image_ref = image.copy()
    image_ref.paste(loading_box_image1 if frame==0 else loading_box_image2, ((240-150)//2 , (416-150)//2))
    return image_ref

@jit(nopython=True,cache = True)
def dump_2bit(pixels):
    pixels = np.clip(pixels, 0, 255)
    pixels_quantized = np.digitize(pixels, bins=[64, 128, 192], right=True)
    
    result_size = (pixels.size + 7) // 8  # Calculate the needed size for the result
    int_pixels = np.zeros(result_size, dtype=np.uint8)
    
    index = 0
    for i in range(pixels_quantized.size):
        bit = 1 if pixels_quantized.flat[i] in [2, 3] else 0
        if i % 8 == 0 and i > 0:
            index += 1
        int_pixels[index] |= bit << (7 - (i % 8))
    return int_pixels


# def image_to_header_file(image):
#     """Apply Floyd-Steinberg dithering and convert image to a string array."""
#     grayscale = image.convert('L')
#     pixels = np.array(grayscale, dtype=np.float32)
#     for y in range(pixels.shape[0]-1):
#         for x in range(1, pixels.shape[1]-1):
#             old_pixel = pixels[y, x]
#             new_pixel = np.round(old_pixel / 85) * 85
#             pixels[y, x] = new_pixel
#             quant_error = old_pixel - new_pixel
#             pixels[y, x+1] += quant_error * 7 / 16
#             pixels[y+1, x-1] += quant_error * 3 / 16
#             pixels[y+1, x] += quant_error * 5 / 16
#             pixels[y+1, x+1] += quant_error * 1 / 16
#     # raw_pixels = pixels.copy()
#     pixels = np.clip(pixels, 0, 255)
#     pixels_quantized = np.digitize(pixels, bins=[64, 128, 192], right=True)
#     pixel_map = {0: '00', 1: '01', 2: '10', 3: '11'} 
#     pixels_string = np.vectorize(pixel_map.get)(pixels_quantized)
#     converted_pixels = pixels_string.flatten().tolist() 
#     # if two_bit : converted_pixels = converted_pixels[::-1]
#     group_size = 4 
#     grouped_pixels = [''.join(converted_pixels[i:i+group_size]) for i in range(0, len(converted_pixels), group_size)]
#     int_pixels = [int(bits, 2) for bits in grouped_pixels] 

#     # return np.array(int_pixels, dtype=np.uint8)
#     return [int(x) for x in int_pixels]

@jit(nopython=True)
def floydSteinbergDithering_numba(pixels):
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
    return pixels

def image_to_header_file(image):
    grayscale = image.convert('L')
    pixels = np.array(grayscale, dtype=np.float32)

    # Apply the Numba-optimized Floyd-Steinberg dithering
    pixels = floydSteinbergDithering_numba(pixels)

    # Convert pixels to 2-bit representation
    pixels = np.clip(pixels, 0, 255)  # Ensure pixels are in valid range after dithering
    pixels_quantized = np.digitize(pixels, bins=[64, 128, 192], right=True)
    pixel_map = {0: '00', 1: '01', 2: '10', 3: '11'} 
    pixels_string = np.vectorize(pixel_map.get)(pixels_quantized).flatten()

    # Convert the binary string to integer values
    group_size = 4
    grouped_pixels = [''.join(pixels_string[i:i+group_size]) for i in range(0, len(pixels_string), group_size)]
    int_pixels = [int(bits, 2) for bits in grouped_pixels]

    return [int(x) for x in int_pixels]