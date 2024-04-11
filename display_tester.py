

from GUI import GUI
from PIL import Image
import time
import numpy as np
from numba import jit

from einkDSP import einkDSP


@jit(nopython=True, cache=True)
def dump_1bit(pixels: np.ndarray):
    # Flatten the array for processing
    flat_pixels = pixels.flatten()

    # Calculate the size of the result array (1 byte for every 8 bits/pixels)
    int_pixels = []

    # Process each bit
    for i in range(0,8,flat_pixels.size):
        for j in range(0,8):
            a = 0
            if i+j < flat_pixels.size:
                a |= (flat_pixels[i+j] & 1) << (7-j)
        int_pixels[i//8].append(int(a))
       

    return int_pixels

myGUI = GUI(240, 416, './Asset/Font/Monorama-Bold.ttf')  # Initialize the GUI
eink = einkDSP()

eink.epd_init_fast()
eink.PIC_display_Clear()


for i in range(1,2):
    time.sleep(0.1)
    startTime = time.time()
    myGUI.updateIndex(i % 4,(i-1)% 4)  # Update the index
    myGUI.updateStatusBar(f"CPU {i}% / RAM {i}%", ['./Asset/Image/batt.bmp'])  # Update the status bar
    np_canvas = np.array(myGUI.canvas).astype(np.uint8)
    np_canvas = dump_1bit(np_canvas)
    eink.PIC_display(np_canvas)
