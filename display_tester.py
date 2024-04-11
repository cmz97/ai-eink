

from GUI import GUI
from PIL import Image
import time
import numpy as np
from numba import jit

from einkDSP import einkDSP

# @jit(nopython=True,cache = True)
# def dump_1bit(pixels):
#     # pixels = np.clip(pixels, 0, 255)
#     # pixels_quantized = np.digitize(pixels, bins=[64, 128, 192], right=True) # 64, 128, 192
#     pixels_quantized = pixels
#     result_size = (pixels.size + 7) // 8  # Calculate the needed size for the result
#     int_pixels = np.zeros(result_size, dtype=np.uint8)
    
#     index = 0
#     for i in range(pixels_quantized.size):
#         bit = 1 if pixels_quantized.flat[i] in [2, 3] else 0
#         if i % 8 == 0 and i > 0:
#             index += 1
#         int_pixels[index] |= bit << (7 - (i % 8))
#     return int_pixels

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

@jit(nopython=True,cache = True)
def dump_1bit(pixels):
    pixels = floydSteinbergDithering_numba(pixels)
    pixels = np.clip(pixels, 0, 255)
    pixels_quantized = np.digitize(pixels, bins=[64, 128, 192], right=True) # 64, 128, 192
    
    result_size = (pixels.size + 7) // 8  # Calculate the needed size for the result
    int_pixels = np.zeros(result_size, dtype=np.uint8)
    
    index = 0
    for i in range(pixels_quantized.size):
        bit = 1 if pixels_quantized.flat[i] in [2, 3] else 0
        if i % 8 == 0 and i > 0:
            index += 1
        int_pixels[index] |= bit << (7 - (i % 8))
    return int_pixels

myGUI = GUI(240, 416, './Asset/Font/Monorama-Bold.ttf')  # Initialize the GUI
eink = einkDSP()

eink.epd_init_part()
eink.PIC_display_Clear()


for i in range(1,2):
    time.sleep(0.1)
    startTime = time.time()
    myGUI.updateIndex(i % 4,(i-1)% 4)  # Update the index
    myGUI.updateStatusBar(f"CPU {i}% / RAM {i}%", ['./Asset/Image/batt.bmp'])  # Update the status bar
    print(f"GUI time: {time.time() - startTime}")
    startTime = time.time()

    np_canvas = np.array(myGUI.canvas).astype(np.uint8)
    np_canvas = dump_1bit(np_canvas).tolist()
    eink.epd_init_part()
    eink.PIC_display(np_canvas)
    print(f"EINK time: {time.time() - startTime}\n")
    eink.PIC_display_Clear()
