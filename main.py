import asyncio
from einkDSP import einkDSP
from encoder import *
from GUI import *

class Page:
    def __init__(self, app):
        self.app = app

    async def display(self):
        """Async method to define how the page displays itself."""
        pass

    async def handle_input(self, input):
        """Async method to handle user input."""
        pass

class HomePage(Page):
    
    def __init__(self, app):
        super().__init__(app)
        self.index = 0
        self.gui = GUI()
        
    async def handle_input(self, input):
        if input == 'up':
            last = self.index
            self.index = (self.index - 1) % len(self.gui.contents)
            self.gui.updateIndex(self.index, last)
        elif input == 'down':
            last = self.index
            self.index = (self.index + 1) % len(self.gui.contents)
            self.gui.updateIndex(self.index, last)
        elif input == 'enter':
            pass
        else:
            pass

# Implement other ProgramPage classes similarly with async methods

class Application:
    def __init__(self):
        self.eink = einkDSP()
        self.current_page = HomePage(self)

        buttons = [
            {'pin': 9, 'direction': 'up', 'callback': self.press_callback},
            {'pin': 22, 'direction': 'down', 'callback': self.press_callback},
            {'pin': 17, 'direction': 'enter', 'callback': self.press_callback}
        ]
        
        self.multi_button_monitor = MultiButtonMonitor(buttons)

    async def einkDisplay(self, hex_pixels):
        self.eink.epd_w21_init_4g()
        self.eink.pic_display_4g(hex_pixels)
        self.eink.epd_sleep()
            
    async def press_callback(self, key):
        self.current_page.handle_input(key)
        

async def main():
    app = Application()
    await app.run()

if __name__ == "__main__":
    asyncio.run(main())
