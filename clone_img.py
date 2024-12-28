from PIL import Image
import os

# Paths
input_folder = "input" # images to clone

# Count of images to clone
amount = 299
images2clone = os.listdir(input_folder)
amount_helper = int(amount / len(images2clone)) # Ensure that the amount is the same if initial image is not single

# Loop through each image in the folder and clone it in total 'amount' times.
for filename in images2clone:
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        input_path = os.path.join(input_folder, filename)

        with Image.open(input_path) as img:
            for i in range(0, amount_helper):
                img.save(input_path.split('.')[0] + f'_{i}.' + input_path.split('.')[1])