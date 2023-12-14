import argparse
import json
import os
from tools.crop_images import crop_images_in_directory
from tools.create_downsamples import scale_images
from tools.separate_train_val_datasets import split_train_val_images

# Specify the directory containing the images
parser = argparse.ArgumentParser(description="Segmentation evaluation script")
parser.add_argument("--json_file_path", type=str, default="/home/fatbardhf/SuperResolution_for_in-field_weed_images/configs/create_training_dataset.json", help="Path to json config file")

args = parser.parse_args()
json_file_path = args.json_file_path

# Read configuration from JSON file
with open(json_file_path, mode='r') as file:
    config = json.load(file)

# Specify the directory to save the cropped images
original_patches_directory = os.path.join(config["training_images_dir"], "1x")

if not os.path.exists(original_patches_directory):
        os.makedirs(original_patches_directory)

# Crop images in the directory and save them 
# Here we take the relevant images and cut them into patches
crop_images_in_directory(config["original_images_dir"], save_dir = original_patches_directory, dataset_portion=config["dataset_portion"] )

scale_images( original_images_dir=original_patches_directory, scaling_factors=config["scaling_factors"],  new_images_dir=config["training_images_dir"])


# Separate in train and val 
for scaling_factor in ( ["1x"] + config["scaling_factors"]):
    # Set the train and validation directory paths
    source_directory = os.path.join(config["training_images_dir"], scaling_factor )
    train_directory = os.path.join(source_directory, "train")
    val_directory = os.path.join(source_directory, "val")

    # Set the random seed
    random_seed = 1234

    # Call the function to split images into train and validation folders
    split_train_val_images(source_directory, train_directory, val_directory, config["val_portion"], random_seed)