

from GUI import GUI
from PIL import Image
import time
import numpy as np
import einkDSP
from numba import jit

from einkDSP import einkDSP


@jit(nopython=True,cache = True)
def dump_2bit(pixels:np.ndarray):

    result_size = (pixels.size + 7) // 8  # Calculate the needed size for the result
    int_pixels = np.zeros(result_size, dtype=np.uint8)
    
    index = 0
    for i in range(pixels.size):
        bit = 1 if pixels.flat[i] in [2, 3] else 0
        if i % 8 == 0 and i > 0:
            index += 1
        int_pixels[index] |= bit << (7 - (i % 8))
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
    eink.PIC_display(dump_2bit(np_canvas))
