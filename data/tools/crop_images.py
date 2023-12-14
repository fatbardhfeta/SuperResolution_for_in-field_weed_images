import os
from PIL import Image
from tqdm import tqdm


def crop_image(image_path, save_dir):
    # Open the image
    image = Image.open(image_path)
    window_size = (256, 256)

    # Get the width and height of the image
    width, height = image.size

    # Calculate the number of rows and columns based on the window size
    rows = (height + window_size[1] - 1) // window_size[1]
    columns = (width + window_size[0] - 1) // window_size[0]

    # Crop the image into rows x columns parts
    for row in range(rows):
        for col in range(columns):
            left = col * window_size[0]
            top = row * window_size[1]
            right = left + window_size[0]
            bottom = top + window_size[1]

            # Check if the current crop exceeds the image boundaries
            right = min(right, width)
            bottom = min(bottom, height)

            cropped_part = image.crop((left, top, right, bottom))

            # Check if the cropped part has the correct window size
            if cropped_part.size == window_size:
                # Create the save directory if it doesn't exist
                os.makedirs(save_dir, exist_ok=True)

                # Save the cropped part
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                part_name = f"{base_name}_part_{row}_{col}.png"
                part_path = os.path.join(save_dir, part_name)
                cropped_part.save(part_path)

def crop_images_in_directory(directory, save_dir, dataset_portion):
    # Find all PNG images in the directory
    png_files = [file for file in os.listdir(directory) if file.endswith(".png")]
    dataset_size = int(len(png_files) * dataset_portion)
    png_files = png_files[:dataset_size]

    # Crop each image in the directory
    for file in tqdm(png_files, desc="Cropping images into patches", unit="image"):
        image_path = os.path.join(directory, file)
        crop_image(image_path, save_dir)

