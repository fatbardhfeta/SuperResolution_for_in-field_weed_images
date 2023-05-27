import os
from PIL import Image
from tqdm import tqdm

# Paths
original_images_dir = "/home/fatbardhf/data_code/data/A_20210707/png_lenscor"
new_images_dir = "/home/fatbardhf/bicubical"

# Create folders for downsampled and upscaled images
os.makedirs(os.path.join(new_images_dir, "2x"), exist_ok=True)
os.makedirs(os.path.join(new_images_dir, "3x"), exist_ok=True)
os.makedirs(os.path.join(new_images_dir, "4x"), exist_ok=True)

# Get the list of original image files
original_images = [filename for filename in os.listdir(original_images_dir)
                   if filename.endswith(".jpg") or filename.endswith(".png")]

# Iterate over original images with progress bar
for filename in tqdm(original_images, desc="Processing images", unit="image"):
    # Load the original image
    original_image_path = os.path.join(original_images_dir, filename)
    original_image = Image.open(original_image_path)

    # Downscale the image
    downscaled_image_2x = original_image.resize((original_image.width // 2, original_image.height // 2), Image.BICUBIC)
    downscaled_image_3x = original_image.resize((original_image.width // 3, original_image.height // 3), Image.BICUBIC)
    downscaled_image_4x = original_image.resize((original_image.width // 4, original_image.height // 4), Image.BICUBIC)


    # Upscale the downscaled images
    upscaled_image_2x = downscaled_image_2x.resize((original_image.width, original_image.height), Image.BICUBIC)
    upscaled_image_3x = downscaled_image_3x.resize((original_image.width, original_image.height), Image.BICUBIC)
    upscaled_image_4x = downscaled_image_4x.resize((original_image.width, original_image.height), Image.BICUBIC)

    # Save the upscaled images
    upscaled_image_2x.save(os.path.join(new_images_dir, "2x", f"2x_{filename}"))
    upscaled_image_3x.save(os.path.join(new_images_dir, "3x", f"3x_{filename}"))
    upscaled_image_4x.save(os.path.join(new_images_dir, "4x", f"4x_{filename}"))

