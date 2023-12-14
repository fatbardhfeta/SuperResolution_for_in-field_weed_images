import os
from PIL import Image

def reconstruct_images_in_directory(directory, save_dir):
    # Find all image files in the directory
    image_files = [file for file in os.listdir(directory) if file.endswith(".png")]

    # Create a dictionary to store the cropped parts
    image_parts = {}

    # Iterate over each image file
    for file in image_files:
        image_path = os.path.join(directory, file)

        # Extract the base name of the image
        base_name = "_".join(file.split("_")[:-4])

        # Extract the row and column indices from the file name
        row, col = int(file.split("_")[-4]), int(file.split("_")[-3])

        # Open the cropped part image
        cropped_part = Image.open(image_path)

        # Add the cropped part to the dictionary
        if base_name not in image_parts:
            image_parts[base_name] = {}

        image_parts[base_name][(row, col)] = cropped_part

    # Reconstruct the original images
    for base_name in image_parts:
        # Get the dimensions of the original image
        max_row = max(image_parts[base_name], key=lambda x: x[0])[0]
        max_col = max(image_parts[base_name], key=lambda x: x[1])[1]

        # Get the dimensions of the cropped parts
        part_width = image_parts[base_name][(0, 0)].width
        part_height = image_parts[base_name][(0, 0)].height

        # Calculate the dimensions of the reconstructed image
        #reconstructed_width = (max_col + 1) * part_width
        #reconstructed_height = (max_row + 1) * part_height
        reconstructed_width = 5464
        reconstructed_height = 3640
        # Create a new image to store the reconstructed image
        reconstructed_image = Image.new("RGB", (reconstructed_width, reconstructed_height))

        # Iterate over each part and paste it into the reconstructed image
        for row in range(max_row + 1):
            for col in range(max_col + 1):
                part = image_parts[base_name].get((row, col))
                if part:
                    left = col * part_width
                    top = row * part_height
                    reconstructed_image.paste(part, (left, top))

        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save the reconstructed image
        save_path = os.path.join(save_dir, f"{base_name}.png")
        reconstructed_image.save(save_path)


# # Specify the directory containing the cropped images
# cropped_images_directory = "/home/fatbardhf/thesis/HAT/results/HAT_SRx2/visualization/flight_altitude"

# # Specify the directory to save the reconstructed images
# save_reconstructed_directory = "/home/fatbardhf/my_procesed_data/hat_upscaled/2x"

# # Reconstruct the images and save them
# reconstruct_images_in_directory(cropped_images_directory, save_reconstructed_directory)
