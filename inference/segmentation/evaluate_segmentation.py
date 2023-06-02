import os
import argparse
from PIL import Image
from tqdm import tqdm
from EWISSeg.predict import predict
import torch
from EWISSeg.utils import convert_labelmap_to_color
from skimage import io as skio
from pathlib import Path
from EWISSeg.dataset import get_loader
import segmentation_models_pytorch as smp
from segment_downscaled import get_slices_per_image, reshape_by_image, combine_labelmap_from_slices
from create_downsamples import scale_images
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

model_suffix = "model2021"
num_img = 330
grid = (22, 15)
device = "cuda"
batch_size = 50

def segment_images(model_path, data_paths):
    
    model_save_path = model_path

    loaded_model = torch.load(model_save_path, map_location=torch.device("cuda"))
    encoder_name = loaded_model["encoder_name"]
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
    )
    model.load_state_dict(loaded_model["model_state_dict"])
    model = model.to(device)

    segmentations = []
    for img_l in data_paths:
        img = skio.imread(img_l)
        generator = torch.Generator()
        generator.manual_seed(42)

        dataloader = get_loader(
            img_ls=[img_l],
            slc_size=256,
            b_crop=False,
            filter_thresh=0,
            split="test",
            generator=generator,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True
        )

        preds = predict(model=model, test_loader=dataloader, device=device)
        preds_labelmaps = get_slices_per_image(preds, grid)
        preds = reshape_by_image(preds_labelmaps, grid, img_shape=img.shape)

        segmentations.append(convert_labelmap_to_color(preds.to("cpu")))
    
    return segmentations

def dice_score(images1, images2):
    dice_scores = []
    for img1, img2 in zip(images1, images2):
        intersection = np.logical_and(img1, img2).sum()
        dice = (2.0 * intersection) / (img1.sum() + img2.sum())
        dice_scores.append(dice)
    return np.mean(dice_scores)

def calculate_dice_scores(predictions):
    original = predictions["original"]
    dice_scores = {}

    for key, value in predictions.items():
        if key != "original":
            dice = dice_score(original, value)
            dice_scores[key] = dice

    return dice_scores

def save_dice_scores_graph(dice_scores, save_dir):
    # Extract the scaling factors and dice scores
    scaling_factors = list(dice_scores.keys())
    scores = list(dice_scores.values())

    # Create a bar plot
    plt.bar(scaling_factors, scores)
    plt.xlabel("Scaling Factors")
    plt.ylabel("Dice Score")
    plt.title("Dice Scores for Different Scaling Factors")
    plt.tight_layout()

    # Save the graph
    graph_path = os.path.join(save_dir, "dice_scores_graph.png")
    plt.savefig(graph_path)
    plt.close()

    print("Dice scores graph saved successfully.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Segmentation evaluation script")
    parser.add_argument("--model_path", type=str, help="Path to the segmentation model")
    parser.add_argument("--images_dir", type=str, help="Path to the images")
    args = parser.parse_args()

    # Paths
    original_images_dir = args.images_dir

    original_images =[os.path.join(original_images_dir, filename) for filename in os.listdir(original_images_dir)
                   if filename.endswith(".jpg") or filename.endswith(".png")]

    #print(original_images)
    model_path = args.model_path
    #scaling_factors = ["2x", "4x", "6x", "8x", "10x", "12x", "15x", "20x", "22x", "25x"]
    scaling_factors = ["2x", "4x", "6x", "8x"]

    downscaled_images_dirs = scale_images(original_images_dir, scaling_factors=scaling_factors)
    
    pred = { 'original' : segment_images( model_path, original_images)} 

    for downscaled_image_folder, index in zip(downscaled_images_dirs, scaling_factors):
        downscaled_images = [os.path.join(downscaled_image_folder, filename) for filename in os.listdir(downscaled_image_folder)
                   if filename.endswith(".jpg") or filename.endswith(".png")]

        pred[index] = segment_images( model_path, downscaled_images)

    dice_scores = calculate_dice_scores(pred)

    print(dice_scores)
    print_predictions_table(dice_scores, "inference/segmentation")
    

main()

