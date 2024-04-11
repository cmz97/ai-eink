

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
    result_size = (flat_pixels.size + 7) // 8
    int_pixels = np.zeros(result_size, dtype=np.uint8)

    # Process each bit
    for i in range(flat_pixels.size):
        # Determine the index in the result array
        index = i // 8
        # Accumulate bits into bytes
        int_pixels[index] |= flat_pixels[i] << (7 - (i % 8))

    return int_pixels.tolist()

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
