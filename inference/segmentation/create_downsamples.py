import os
from PIL import Image
from tqdm import tqdm

def scale_images(original_images_dir, scaling_factors, new_images_dir = "inference/segmentation/results"):
    # Create folders for downsampled and upscaled images
    new_images_dirs = []
    for factor in scaling_factors:
        new_images_factor_dir = os.path.join(new_images_dir, factor)
        os.makedirs(new_images_factor_dir, exist_ok=True)
        new_images_dirs.append(new_images_factor_dir)

    # Get the list of original image files
    original_images = [filename for filename in os.listdir(original_images_dir)
                       if filename.endswith(".jpg") or filename.endswith(".png")]

    # Iterate over original images with progress bar
    for filename in tqdm(original_images, desc="Downsampling images", unit="image"):
        # Load the original image
        original_image_path = os.path.join(original_images_dir, filename)
        original_image = Image.open(original_image_path)

        for factor, new_images_factor_dir in zip(scaling_factors, new_images_dirs):
            # Calculate the scaling factor
            scale = int(factor.replace("x", ""))
            # Downscale the image
            downscaled_image = original_image.resize((original_image.width // scale, original_image.height // scale), Image.BICUBIC)

            # Upscale the downscaled image
            upscaled_image = downscaled_image.resize((original_image.width, original_image.height), Image.BICUBIC)

            # Save the upscaled image
            upscaled_image.save(os.path.join(new_images_factor_dir, f"{factor}_{filename}"))

    return new_images_dirs
