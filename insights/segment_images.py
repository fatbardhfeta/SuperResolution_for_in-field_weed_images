import os
import argparse
import json
import torch
import segmentation_models_pytorch as smp
from tools.evaluation_functions import segment_images
from pathlib import Path


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Segmentation evaluation script")
    parser.add_argument("--json_file_path", type=str, default="/home/fatbardhf/SuperResolution_for_in-field_weed_images/configs/segmentaion_configuration.json", help="Path to json config file")
    args = parser.parse_args()

    # Read configuration from JSON file
    with open(args.json_file_path, mode='r') as file:
        config = json.load(file)

    #model_suffix = config["model_suffix"]
    model_path = config["model_path"]
    #batch_size = config["batch_size"]

    experiments = config["experiments"]

    device = config["device"]


    for experiment in experiments:
        experiment_name = experiment["experiment_name"]
        print(f"Runing segmentation for : {experiment_name}")
        
        scaling_factors = experiment["scaling_factors"]
        upscaled_images_path = experiment["upscaled_images_path"]

        # Load segmentation model
        loaded_model = torch.load(model_path, map_location=torch.device(device))
        model = smp.Unet(
            encoder_name=loaded_model["encoder_name"],
            encoder_weights="imagenet",
            in_channels=3,
            classes=3,
        ).to(device)
        model.load_state_dict(loaded_model["model_state_dict"])

        pred = {}
        for index in scaling_factors:
            upscaled_image_dir = os.path.join(upscaled_images_path, index)
            pred[index] = segment_images(model, Path(upscaled_image_dir), options=config)


if __name__ == "__main__":
    main()
