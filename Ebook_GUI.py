import time
from PIL import Image,ImageDraw,ImageFont, ImageOps
import numpy as np
from numba import jit
import textwrap

# self.eink_width, self.eink_height = 240, 416

class EbookGUI:
    def __init__(self, eink_width=240, eink_height=416, FONT_PATH='./Asset/Font/Monorama-Bold.ttf', icons = ['./Asset/Image/batt.bmp']):
        self.eink_width, self.eink_height = eink_width, eink_height
        self.FONT_PATH = './Asset/Font/Monorama-Bold.ttf'

        self.contents = ""

        self.canvas = Image.new('1', (eink_width, eink_height), 'white')
        self.draw_plus_pattern(self.canvas, density=10, start_pos=(5, 0), size=5)  # Draw '+' pattern
        self.canvas = self.draw_status_bar_with_text_and_icons(self.canvas, "CPU 100% / RAM 100%", icons, 35, 5, FONT_PATH, 15)


        itemBox, _ = self.draw_rounded_rectangle_with_mask(eink_width, eink_height, 10, (10, 10, 10, 10), 3, fill=False)
        top_left = (0, 45)
        self.canvas.paste(itemBox, top_left)
        self.text_area = ((top_left[0] + 10, top_left[1] + 10), (eink_width - 10, eink_height - 10))

        self.font_size = 15

        
    def draw_plus_pattern(self,canvas, density, start_pos, size):
        """
        Draw a '+' pattern on the canvas starting from start_pos, with specified size and density.

        :param canvas: PIL Image object where the '+' pattern will be drawn.
        :param density: Determines how close each '+' is to its neighbor.
        :param start_pos: A tuple (x, y) indicating the starting position for the pattern.
        :param size: The size of the '+', which also dictates the width of each stroke. Minimum value is 3.
        """
        draw = ImageDraw.Draw(canvas)
        width, height = canvas.size
        x_start, y_start = start_pos

        # Ensure size is at least 3
        size = max(size, 3)
        half_size = size // 2

        # Calculate the step between each '+' based on density
        step = max(size, density)

        for y in range(y_start, height, step):
            for x in range(x_start, width, step):
                # Vertical line of '+'
                draw.line([(x, y - half_size), (x, y + half_size)], fill='black')
                # Horizontal line of '+'
                draw.line([(x - half_size, y), (x + half_size, y)], fill='black')
   
    def updateStatusBar(self, text, icons):
        self.canvas = self.draw_status_bar_with_text_and_icons(self.canvas, text, icons, 35, 5, self.FONT_PATH, 15)
        self.canvas.save('result.bmp')

    # --- Main script ---
    def draw_rounded_rectangle_with_mask(self, width, height, corner_radius, padding, line_thickness, fill):
        # Similar setup as before
        image = Image.new('1', (width, height), 'white')
        self.draw_plus_pattern(image, density=10, start_pos=(5, 5), size=5)  # Draw '+' pattern

        mask = Image.new('1', (width, height), 'black')  # Black mask, white for areas to keep
        draw_image = ImageDraw.Draw(image)
        draw_mask = ImageDraw.Draw(mask)
        
        # Adjust the drawing rectangle for the padding
        padded_rectangle = (padding[3], padding[0], width - padding[2], height - padding[1])
        
        # Draw the rounded rectangle on both image and mask
        fill_color = 'black' if fill else 'white' 
        draw_image.rounded_rectangle(padded_rectangle, radius=corner_radius, fill=fill_color, outline='black', width=line_thickness)
        draw_mask.rounded_rectangle(padded_rectangle, radius=corner_radius, fill='white',  outline='black', width=line_thickness)

        return image, mask



    def draw_text_on_canvas(self, canvas, wrapped_lines):
        """
        Draw text on a canvas, handling newline characters and wrapping text within the canvas width.
        """
        font_path = self.FONT_PATH
        font_size = self.font_size
        box = self.text_area
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(font_path, font_size)
        canvas_width, canvas_height = canvas.size
        text_area_start, text_area_end = box
        y_text = text_area_start[1]
        
        for line in wrapped_lines:
            # Draw each line of text
            draw.text((text_area_start, y_text), line, fill=0, font=font)
            y_text += font_size * 1.2  # Increment y position by line height
        
        return canvas


    def create_row_layout(self, canvas, img_path, text, font_path, font_size, padding):
        """
        Create a row layout inside a rounded rectangle, with an image on the left and text on the right, 
        correctly handling newline characters.
        """
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(font_path, font_size)
        
        # Image processing
        img = Image.open(img_path).convert('1')  # Convert image to 1-bit black and white
        aspect_ratio = img.width / img.height
        canvas_width, canvas_height = canvas.size
        target_height = canvas_height - padding[0] - padding[1]
        target_width = int(aspect_ratio * target_height)
        img_resized = img.resize((target_width, target_height))
        
        # Place image
        canvas.paste(img_resized, (padding[3], padding[0]))
        
        available_width = canvas_width - padding[2] - padding[3] - target_width
        text_area_start = target_width + padding[3] + 5 # Add a small gap between image and text
        
        y_text = padding[0]

        # Handle new lines explicitly and wrap text within each segment
        segments = text.split('\n')  # Split text by newline characters
        for segment in segments:
            wrapped_lines = textwrap.wrap(segment, width=30, max_lines=None, 
                                        placeholder="…", 
                                        break_long_words=True, 
                                        break_on_hyphens=True)
            for line in wrapped_lines:
                # Draw each line of text
                draw.text((text_area_start, y_text), line, fill=0, font=font)
                y_text += font_size * 1.2  # Increment y position by line height

        return canvas

    def invert_pixels_within_region(self, image, mask, top_left):
        """
        Inverts pixels within a specified region of a binary image using a mask.
        Assumes image is in '1' (binary) mode and mask is correctly aligned.
        
        :param image: The PIL Image object to modify, in '1' mode.
        :param mask: A PIL Image object representing the mask, same size as the region.
        :param top_left: A tuple (x, y) specifying the top left corner of the region.
        """
        # Ensure working with 'L' mode for array operations
        image = image.convert('L')
        mask = mask.convert('L')

        # Convert to NumPy arrays
        image_array = np.array(image, dtype=np.uint8)
        mask_array = np.array(mask, dtype=np.uint8)

        x_start, y_start = top_left
        x_end, y_end = x_start + mask.width, y_start + mask.height

        # Select the region to apply the mask
        region = image_array[y_start:y_end, x_start:x_end]

        # Invert the region based on the mask
        # Pixels in the region are inverted where the mask is white
        region_inverted = np.where(mask_array == 255, 255 - region, region)

        # Place the inverted region back
        image_array[y_start:y_end, x_start:x_end] = region_inverted

        # Convert back to PIL Image in '1' mode
        return Image.fromarray(image_array, 'L').convert('1')

    def process_icon_for_status_bar(self, icon_path, new_width, new_height):
        """
        Load an icon, resize it, and convert white pixels to transparent and black pixels to white.

        :param icon_path: Path to the icon image.
        :param new_width: New width after resizing.
        :param new_height: New height after resizing.
        :return: Processed PIL Image object.
        """
        icon = Image.open(icon_path).convert("RGBA")
        icon = icon.resize((new_width, new_height))
        
        # Create a mask where white pixels are treated as transparent
        mask = Image.new("L", icon.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        for x in range(icon.width):
            for y in range(icon.height):
                r, g, b, a = icon.getpixel((x, y))
                if r > 200 and g > 200 and b > 200:  # Assuming white pixels
                    mask_draw.point((x, y), 0)  # Transparent in mask
                else:
                    mask_draw.point((x, y), 255)  # Opaque in mask
                    icon.putpixel((x, y), (255, 255, 255, 255))  # Convert black pixels to white
        
        # Apply the mask to the icon
        icon.putalpha(mask)
        
        return icon

    def draw_status_bar_with_text_and_icons(self, screen, text, icons, bar_height, padding, font_path, font_size):
        """
        Draws a status bar on top of the screen with one text box at the start and two icons to the right,
        equally spaced and resized proportionally to fit within the status bar.

        :param screen: PIL Image object representing the screen where the status bar will be drawn.
        :param text: Text to display in the text box.
        :param icons: List of paths to the icon images (length should be 2).
        :param bar_height: Height of the status bar.
        :param padding: Padding inside the status bar around the text box and icons.
        :param font_path: Path to the font file.
        :param font_size: Size of the font.
        """
        draw = ImageDraw.Draw(screen)
        font = ImageFont.truetype(font_path, font_size)
        screen_width = screen.size[0]

        # Draw the rounded rectangle for the status bar
        draw.rounded_rectangle([(0, -10), (screen_width, bar_height)], radius=10, fill="black")

        # Text box handling with newline characters
        text_area_start = padding + 5
        y_text = padding + 6  # Add a small vertical padding
        segments = text.split('\n')  # Split text by newline characters
        for segment in segments:
            wrapped_lines = textwrap.wrap(segment, width=30, max_lines=None,
                                        placeholder="…", break_long_words=True, break_on_hyphens=True)
            for line in wrapped_lines:
                draw.text((text_area_start, y_text), line, fill="white", font=font)
                y_text += int(font_size * 1.2)  # Increment y position by line height
        
        text_area_end = text_area_start + max(draw.textlength(line, font=font) for line in wrapped_lines) + padding
        
        # Resize and place icons
        icon_space = int((screen_width - text_area_end - padding * (len(icons) + 1)) // len(icons))
        for i, icon_path in enumerate(icons):
            icon = Image.open(icon_path)
            aspect_ratio = icon.width / icon.height
            new_height = bar_height - 2 * padding
            new_width = int(aspect_ratio * new_height)

            # Process icon (resize and convert colors)
            icon = self.process_icon_for_status_bar(icon_path, new_width, new_height)

            x_position = int(190+ icon_space * i)
            y_position = int((bar_height - new_height) // 2)
            screen.paste(icon, (x_position, y_position), icon)
        
        return screen
    


# myGUI = GUI(240, 416, './Asset/Font/Monorama-Bold.ttf')  # Initialize the GUI

# for i in range(1,100):
#     time.sleep(0.1)
#     startTime = time.time()
#     myGUI.updateIndex(i % 4,(i-1)% 4)  # Update the index
#     myGUI.updateStatusBar(f"CPU {i}% / RAM {i}%", ['./Asset/Image/batt.bmp'])  # Update the status bar
#     print(f"Time taken: {time.time() - startTime:.4f} seconds")
