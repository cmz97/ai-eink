

from GUI import GUI
from PIL import Image
import time
import numpy as np
from numba import jit
import sys
from einkDSP import einkDSP
# np.set_printoptions(threshold=sys.maxsize)


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

@jit(nopython=True, cache=True)
def dump_1bit(pixels: np.ndarray):
    # Flatten the array for processing
    flat_pixels = pixels.flatten()

    # Calculate the size of the result array (1 byte for every 8 bits/pixels)
    result_size = (flat_pixels.size + 7) // 8
    int_pixels = np.zeros(result_size, dtype=np.uint8)

    # Process each bit
    for i in range(flat_pixels.size):
        # Determine the index in the result array
        index = i // 8
        # Accumulate bits into bytes
        int_pixels[index] |= flat_pixels[i] << (7 - (i % 8))

    # Convert the NumPy array to a Python list of integers
    return [int(x) for x in int_pixels]

myGUI = GUI(240, 416, './Asset/Font/Monorama-Bold.ttf')  # Initialize the GUI
eink = einkDSP()


def clear_screen():
    # self.eink.PIC_display_Clear()
    image = Image.new("L", (240, 416), "white")
    pixels = dump_1bit(np.array(image, dtype=np.uint8))
    eink.epd_init_part()
    eink.PIC_display(pixels)
    eink.PIC_display_Clear()

eink.epd_init_part()
clear_screen()

for i in range(1,2):
    time.sleep(0.1)
    clear_screen()
    startTime = time.time()
    myGUI.updateIndex(i % 4,(i-1)% 4)  # Update the index
    myGUI.updateStatusBar(f"CPU {i}% / RAM {i}%", ['./Asset/Image/batt.bmp'])  # Update the status bar
    print(f"GUI time: {time.time() - startTime}")
    startTime = time.time()
    img = myGUI.canvas.copy()
    img.transpose(Image.FLIP_TOP_BOTTOM)
    np_canvas = np.array(img).astype(np.uint8)
    np_canvas = dump_1bit(np_canvas)
    # print(np_canvas)
    eink.epd_init_part()
    eink.PIC_display(np_canvas)
    time.sleep(0.5)
    print(f"EINK time: {time.time() - startTime}\n")
