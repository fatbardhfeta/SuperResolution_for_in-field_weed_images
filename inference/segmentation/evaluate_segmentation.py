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
from sklearn.metrics import f1_score
from skimage.metrics import peak_signal_noise_ratio, structural_similarity



model_suffix = "model2021"
num_img = 330
grid = (22, 15)
device = "cuda"
batch_size = 50

using_original_masks = True

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

        #print(preds.shape)

        for pred, fname in zip(preds, [img_l]):
            #con_labelmap_to_color = convert_labelmap_to_color(pred.to("cpu"))
            # skio.imsave("inference/segmentation/temp.png", con_labelmap_to_color, check_contrast=False)
            # print("i")
            # print(con_labelmap_to_color.shape)
            
            segmentations.append(pred.to("cpu"))
            #segmentations.append(convert_labelmap_to_color(pred.to("cpu")))

        
        
    return segmentations


def dice_score(images1, images2, using_original_masks=True):
    dice_scores = []
    for img1, img2 in zip(images1, images2):
        # Convert RGB images to binary masks
        if using_original_masks:
            mask1 = convert_image_to_mask(img1)
        else:
            mask1 = img1   
        mask2 = img2

        #mask2 = convert_image_to_mask(img2)
        #remove background
        
        # Flatten the binary masks
        mask1_flat = mask1.flatten()
        mask2_flat = mask2.flatten()
        
        # Compute the Dice score using the F1 score from scikit-learn
        dice = f1_score(mask1_flat, mask2_flat, average=None, labels=[1, 2])
    
        dice_scores.append(np.mean(dice))
        #print(dice)
    print(dice_scores)
    return np.mean(dice_scores)

# def convert_image_to_mask(image):
#     labels = np.array([(199, 199, 199), (31, 119, 180), (255, 127, 14)])
#     lookup_table = np.array(labels)
#     mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
#     for i, label in enumerate(labels):
#         mask[np.all(image == label, axis=-1)] = i + 1

#     print(mask.shape)
#     return mask

def convert_image_to_mask(rgb_mask):
    """
    encodes 3D RGB Mask into 2D array based on a List of RGB tuples
    """
    labels = np.array([(199, 199, 199), (31, 119, 180), (255, 127, 14)])
    label_map = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
    for idx, label in enumerate(labels):
        label_map[(rgb_mask==label).all(axis=2)] = idx
    
    return label_map



def calculate_dice_scores(predictions, using_original_masks = True):
    original = predictions["original"]
    dice_scores = {}

    for key, value in predictions.items():
        if key != "original":
            dice = dice_score(original, value, using_original_masks)
            dice_scores[key] = dice

    return dice_scores

def calculate_psnr_ssim(predictions, using_original_masks = True):
    original = predictions["original"]
    dice_scores = {}

    for key, value in predictions.items():
        if key != "original":
            dice = calculate_psnr_ssim(original, value, using_original_masks)
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


def save_first_segmentation_as_png(segmentations_dict, save_path):
    """
    Saves the first segmentation image from a dictionary as a PNG file.

    Args:
        segmentations_dict (dict): Dictionary of segmentations where the key is the identifier and the value is a list of segmentation images.
        save_path (str): Path to save the first segmentation image as a PNG file.
    """
    # Get the first segmentation image
    first_segmentation = segmentations_dict["8x"][0]
    #print(first_segmentation)
    # Convert the segmentation image to a PIL image
    #skio.imsave( save_path, first_segmentation,check_contrast=False)
    skio.imsave( save_path, convert_labelmap_to_color(first_segmentation),check_contrast=False)

    # Save the first segmentation as a PNG file
    #first_segmentation_pil.save(save_path, "PNG")
    print(f"First segmentation image saved successfully at: {save_path}")

def load_masks(directory):
    masks = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            mask_path = os.path.join(directory, filename)
            mask = skio.imread(mask_path)
            masks.append(mask)
    return masks


def calculate_psnr_ssim(original_image, downsampled_image):
    psnr = peak_signal_noise_ratio(original_image, downsampled_image)
    ssim = structural_similarity(original_image, downsampled_image, channel_axis = 2, multichannel=True)

    return psnr, ssim

def main():
    # Parse command-line arguments
    
    parser = argparse.ArgumentParser(description="Segmentation evaluation script")
    parser.add_argument("--model_path", type=str, help="Path to the segmentation model")
    parser.add_argument("--images_dir", type=str, help="Path to the images")
    parser.add_argument("--masks_dir", type=str, help="Path to the masks")
    args = parser.parse_args()

    # Paths
    original_images_dir = args.images_dir
    masks_dir = args.masks_dir

    original_images =[os.path.join(original_images_dir, filename) for filename in sorted(os.listdir(original_images_dir))
                   if filename.endswith(".jpg") or filename.endswith(".png")]

    #print(original_images)
    model_path = args.model_path
    scaling_factors = ["1x", "2x", "4x", "6x", "8x", "10x", "12x", "14x", "16x", "18x", "20x", "22x", "25x"]
    #scaling_factors = [ "8x","25x"]

    # Creating the downscaled image folders
    downscaled_images_dirs = scale_images(original_images_dir, scaling_factors=scaling_factors)

    if masks_dir:
        pred = {'original' : load_masks( masks_dir)}
        using_original_masks = True 
        print("Comparing with original mask.")
    else:
        pred = {'original' : segment_images( model_path, original_images)} 
        using_original_masks = False
        print("Comparing with segmentet mask.")
    print(using_original_masks)

    for downscaled_image_folder, index in zip(downscaled_images_dirs, scaling_factors):

        downscaled_images = [os.path.join(downscaled_image_folder, filename) for filename in sorted(os.listdir(downscaled_image_folder))
                   if filename.endswith(".jpg") or filename.endswith(".png")]

        pred[index] = segment_images( model_path, downscaled_images)

    dice_scores = calculate_dice_scores(pred, using_original_masks)

    print(dice_scores)
    save_dice_scores_graph(dice_scores, "inference/segmentation")

    save_first_segmentation_as_png(pred, "inference/segmentation/example_seg.png")
    



main()

