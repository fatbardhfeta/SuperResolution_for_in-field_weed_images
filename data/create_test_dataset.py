import argparse
import json
import os
from tools.crop_images import crop_images_in_directory
from tools.create_downsamples import scale_images, downscale_and_upscale_bicubicaly

# Specify the directory containing the images
parser = argparse.ArgumentParser(description="Prepare data for testing script")
parser.add_argument("--json_file_path", type=str, default="/home/fatbardhf/SuperResolution_for_in-field_weed_images/configs/create_test_dataset.json", help="Path to json config file")

args = parser.parse_args()
json_file_path = args.json_file_path

# Read configuration from JSON file
with open(json_file_path, mode='r') as file:
    config = json.load(file)


if config["type"] == 'downscale':
    scale_images(config["original_test_images_dir"], config["scaling_factors"], new_images_dir = config["new_test_images_dir"])
else:
    downscale_and_upscale_bicubicaly(config["original_test_images_dir"], config["scaling_factors"], new_images_dir = config["new_test_images_dir"])

