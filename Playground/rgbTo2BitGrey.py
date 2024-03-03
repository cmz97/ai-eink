from PIL import Image
import numpy as np

def convert_to_2bit_normal(image):
    """Convert image to 2-bit using normal conversion."""
    # Convert to grayscale
    grayscale = image.convert("L")
    # Map to nearest of the 4 levels
    np_image = np.array(grayscale)
    quantized = np.digitize(np_image, bins=[64, 128, 192], right=True)
    # Convert 4 levels back to 0-255
    np_image_2bit = quantized * 85
    return Image.fromarray(np_image_2bit.astype(np.uint8))

def floyd_steinberg_dithering(image):
    """Apply Floyd-Steinberg dithering to convert image to 2-bit."""
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
    return Image.fromarray(pixels.astype(np.uint8))

def combine_images(image1, image2):
    """Combine two images side by side."""
    total_width = image1.width + image2.width
    max_height = max(image1.height, image2.height)
    combined = Image.new('L', (total_width, max_height))
    combined.paste(image1, (0, 0))
    combined.paste(image2, (image1.width, 0))
    return combined

# Load the original image
original_image = Image.open('/Users/chengmingzhang/Documents/eagle/projects/ai-eink/Asset/A.png')

# Convert using normal method
image_normal = convert_to_2bit_normal(original_image)
image_normal.save('image_normal.bmp')

# Convert using dithering
image_dithered = floyd_steinberg_dithering(original_image)
image_dithered.save('image_dithered.bmp')

# Combine and save
combined_image = combine_images(image_normal, image_dithered)
combined_image.save('combined_image.bmp')
