import asyncio
from einkDSP import einkDSP
from encoder import *
from GUI import *
from utils import *
import subprocess  # Import subprocess module

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Page:
    def __init__(self, app):
        self.app = app

    async def display(self):
        """Async method to define how the page displays itself."""
        pass

    async def handle_input(self, input):
        """Async method to handle user input."""
        pass

    async def eink_init(self):
        pass

class HomePage(Page):
    
    def __init__(self, app):
        super().__init__(app)
        self.index = 0
        self.gui = GUI()
        self.display()
        self.subprogram = ['./app_tiny_diffusion.py', './app_ebook.py', './app_tamago.py', './app_ebook_kid.py']  # Add all your scripts here
        
    def handle_input(self, input):
        logging.info(f"handle_input {input}")
        if input == 'up':
            last = self.index
            self.index = (self.index - 1) % len(self.gui.contents)
            self.gui.updateIndex(self.index, last)
            self.display()
        elif input == 'down':
            last = self.index
            self.index = (self.index + 1) % len(self.gui.contents)
            self.gui.updateIndex(self.index, last)
            self.display()
        elif input == 'enter':
            if 0 <= self.index < len(self.subprogram):
                script_to_run = self.subprogram[self.index]
                logging.info(f'Executing script: {script_to_run}')
                # Run the script and wait for it to complete
                # self.app.multi_button_monitor.stop_monitoring()
                self._print_text("opening...")
                process = subprocess.Popen(['venv/bin/python', script_to_run],)
                process.wait()  # Wait for the subprocess to complete       
                # self.app.multi_button_monitor.start_monitoring()
                print("Sub-Program completed !!!")
                self.eink_init()
                self.display()
        else:
            pass
    
    def display(self):
        logging.info('home page display called')
        hex_pixels = dump_1bit(np.array(self.gui.canvas.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.uint8))
        self.app.eink_display_2g(hex_pixels)

    def _print_text(self, text):
        image = fast_text_display(self.gui.canvas, text)
        hex_pixels = dump_1bit(np.array(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.uint8))
        self.app.eink_display_2g(hex_pixels)

# Implement other ProgramPage classes similarly with async methods

class Application:
    def __init__(self):
        self.eink = einkDSP()
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()
        self.in_4g = False
        self.current_page = HomePage(self)

        buttons = [
            {'pin': 9, 'direction': 'up', 'callback': self.press_callback},
            {'pin': 22, 'direction': 'down', 'callback': self.press_callback},
            {'pin': 17, 'direction': 'enter', 'callback': self.press_callback}
        ]        

        self.multi_button_monitor = MultiButtonMonitor(buttons)

    def eink_display_4g(self, hex_pixels):
        logging.info('eink_display_4g')
        self.eink.epd_w21_init_4g()
        self.eink.pic_display_4g(hex_pixels)
        self.eink.epd_sleep()
        self.in_4g = True
    
    def eink_init(self):
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()

    def eink_display_2g(self, hex_pixels):
        logging.info('eink_display_2g')
        if self.in_4g : 
            self.transit()
            self.in_4g = False

        self.eink.epd_init_part()
        self.eink.PIC_display(hex_pixels)

    def transit(self):
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()
        
    def press_callback(self, key):
        self.current_page.handle_input(key)

async def main():
    app = Application()
    while True:
        print("Main.py Pring...")
        await asyncio.sleep(1)  # Sleep for a bit to yield control to the event loop

if __name__ == "__main__":
    asyncio.run(main())
