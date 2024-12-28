import os
import shutil
from PIL import Image
import numpy as np
#import cupy as cp
from numba import jit

# Timer
import time
start_exe = time.time()

# Paths
input_folder = "input"      # images to process
output_folder = "output"    # to store scaled images
done_folder = "done"        # to store processed images

# Ensure output and done folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(done_folder, exist_ok=True)

# JIT-accelerated image scaling function
@jit(nopython=True) # 300 pics in 6 sec vs 202 sec without JIT decorator
def scale_image(image_array, scale):
    old_height, old_width, channels = image_array.shape
    new_height, new_width = int(old_height * scale), int(old_width * scale)
    scaled_array = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            src_y = int(y / scale)
            src_x = int(x / scale)
            scaled_array[y, x] = image_array[src_y, src_x]

    return scaled_array

# Process images
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        input_path = os.path.join(input_folder, filename)
        
        try:
            # Open and process the image
            with Image.open(input_path) as img:
                image_array = np.array(img)
                scaled_array = scale_image(image_array, 2)
                scaled_image = Image.fromarray(scaled_array)

                # Save scaled image
                output_path = os.path.join(output_folder, filename)
                scaled_image.save(output_path)

            # Move original image to "done" folder
            done_path = os.path.join(done_folder, filename)
            shutil.move(input_path, done_path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# End timer
print(int(time.time() - start_exe))