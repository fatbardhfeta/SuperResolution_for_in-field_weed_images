import torch
import os
import matplotlib.pyplot as plt

import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm

from rich import print
from skimage import io as skio
from PIL import Image
from .plate_data_loader import get_patches_loader
from .EWISSeg.predict import predict
from .EWISSeg.utils import convert_labelmap_to_color
from .dice_score import DiceCalculator

from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def segment_images(model, folder_path, options):

    segmentations = []
    print( f"Segmenting images from: {folder_path}" )
    generator = torch.Generator()
    generator.manual_seed(42)

    dataloader = get_patches_loader(
        data_folder=folder_path,
        #slc_size=256,
        #b_crop=False,
        filter_thresh=0,
        split="val",
        generator=generator,
        batch_size=options["batch_size"],
        num_workers=0,
        pin_memory=True
    )
    preds = predict(model=model, test_loader=dataloader, device=options["device"])


    for pred in preds:
        segmentations.append(pred.to("cpu"))

    image_paths = sorted(os.listdir(folder_path / "images"))

    save_path = folder_path / "segmentations"
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for pred, fname in zip(preds, image_paths):
                #model_suffix = options["model_suffix"]'
                #print(fname)
                skio.imsave(
                    f"{str(save_path)}/{fname.split('/')[-1].split('.')[0]}.png",
                    convert_labelmap_to_color(pred.to("cpu")),
                    check_contrast=False
                ) 

                
                
        
    return segmentations

def dice_score(images1, images2, key=None, using_original_masks=True):
    dice_scores = []
    pixel_values = []
    num_images = len(images1)
    
    for img1, img2 in tqdm(zip(images1, images2), total=num_images, desc=key):
        # Convert RGB images to binary masks
        if using_original_masks:
            mask1 = img1
            mask1[mask1 == 127] = 1
            mask1[mask1 == 255] = 2

            mask2 = convert_image_to_mask(img2)
        #mask2 = img2

        mask1_tensor = torch.from_numpy(mask1).unsqueeze(0)
        mask2_tensor = torch.from_numpy(mask2).unsqueeze(0)

        dc = DiceCalculator(gt=mask1_tensor.clone().detach().to(torch.int64),
                            pred=mask2_tensor.clone().detach().to(torch.int64),
                            device="cpu",n_classes=3)
        dice_scores.append(dc.dice_score[0][1:])

    array_2d = np.vstack(dice_scores)
    #mean_columns = np.nanmean(array_2d, axis=0)

    #return np.mean(mean_columns)
    return array_2d

def inspect_masks_size(images1, key):
    pixel_values = []
    for img1 in images1:
        # Convert RGB images to binary masks
        if key == 'original':
            mask1 = img1
            mask1[mask1 == 127] = 1
            mask1[mask1 == 255] = 2
        else:
            mask1 = convert_image_to_mask(img1)

        mask1_tensor = torch.from_numpy(mask1).unsqueeze(0)

        element_counts1_0 = torch.sum(torch.eq(mask1_tensor, 0)).item()  # Count occurrences of 0
        element_counts1_1 = torch.sum(torch.eq(mask1_tensor, 1)).item()  # Count occurrences of 1
        element_counts1_2 = torch.sum(torch.eq(mask1_tensor, 2)).item()  # Count occurrences of 2

        pixel_values.append([element_counts1_0, element_counts1_1, element_counts1_2])

    return np.asarray(pixel_values)

def calculate_dice_scores(predictions, using_original_masks = True):
    original = predictions["original"]
    dice_scores = {}      #create_psnr_ssim_graph(psnr_ssim, "inference/segmentation")
        #print(psnr_ssim)
        # print(dice_scores)

    for key, value in predictions.items():
        if key != "original":
            dice = dice_score(original, value, key, using_original_masks)
            dice_scores[key] = dice

    return dice_scores

def save_dice_scores_graph(dice_scores, save_dir):
    # Extract the scaling factors and dice scores
    scaling_factors = list(dice_scores.keys())
    scores = list(dice_scores.values())

    # Convert scores to a NumPy array
    scores = np.array(scores)

    # Calculate the mean scores
    mean_scores = np.mean(scores, axis=1)

    # Create the first bar plot for mean scores
    plt.bar(scaling_factors, mean_scores)
    plt.xlabel("Scaling Factors")
    plt.ylabel("Mean Dice Score")
    plt.title("Mean Dice Scores for Different Scaling Factors")
    plt.tight_layout()

    # Add text annotations to the bars
    for i, score in enumerate(mean_scores):
        plt.text(i, score, str(round(score, 2)), ha='center', va='bottom')

    # Save the first graph
    mean_graph_path = os.path.join(save_dir, "mean_dice_scores_graph.png")
    plt.savefig(mean_graph_path)
    plt.close()

    # Create the second bar plot for both values as a group
    plt.bar(np.arange(len(scaling_factors)), scores[:, 0], width=0.4, label='Maize')
    plt.bar(np.arange(len(scaling_factors)) + 0.4, scores[:, 1], width=0.4, label='Weed')
    plt.xlabel("Scaling Factors")
    plt.ylabel("Dice Scores")
    plt.title("Dice Scores for Different Scaling Factors")
    plt.xticks(np.arange(len(scaling_factors)), scaling_factors)  # Adjusted x-ticks
    plt.legend()
    plt.tight_layout()

    # Add text annotations to the bars
    for i, (score1, score2) in enumerate(scores):
        plt.text(i, score1, str(round(score1, 2)), ha='center', va='bottom')
        plt.text(i + 0.4, score2, str(round(score2, 2)), ha='center', va='bottom')

    # Save the second graph
    group_graph_path = os.path.join(save_dir, "dice_scores_group_graph.png")
    plt.savefig(group_graph_path)
    plt.close()

    print("Dice scores graphs saved successfully.")

def convert_image_to_mask(rgb_mask):
    """
    encodes 3D RGB Mask into 2D array based on a List of RGB tuples
    """
    #199,199,199 gri
    #32,119,180 bojqielli
    #255,127,14 portokalli
    labels = np.array([(199, 199, 199), (31, 119, 180), (255, 127, 14)])
    label_map = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
    for idx, label in enumerate(labels):
        label_map[(rgb_mask==label).all(axis=2)] = idx
    
    return label_map

def calculate_images_psnr_ssim(predictions):
    original = predictions["original"]
    dice_scores = {}

    for key, value in predictions.items():
        if key != "original":
            dice = calculate_psnr_ssim(original, value, key)
            dice_scores[key] = dice

    return dice_scores

def calculate_psnr_ssim(original_images, restored_images, key):
    psnr = []
    ssim = []
    num_images = len(original_images)
    
    for original_path, downsampled_path in tqdm(zip(original_images, restored_images), total=num_images, desc=key):
        original_image = np.array(Image.open(original_path))
        downsampled_image = np.array(Image.open(downsampled_path))
        psnr.append(peak_signal_noise_ratio(original_image, downsampled_image))
        ssim.append(structural_similarity(original_image, downsampled_image, channel_axis=2, multichannel=True))

    #return {'psnr': np.mean(psnr), 'ssim': np.mean(ssim)}
    return {'psnr': psnr, 'ssim': ssim}

def save_first_segmentation_as_png(segmentations_dict, save_path):
    first_segmentation = segmentations_dict["1x"][0]
    skio.imsave( save_path, convert_labelmap_to_color(first_segmentation),check_contrast=False)
    print(f"First segmentation image saved successfully at: {save_path}")

def load_masks(directory):
    masks = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            mask_path = os.path.join(directory, filename)
            mask = skio.imread(mask_path)
            masks.append(mask)
    return masks

def sort_filenames_by_integer_value(directory):
    # Get the list of filenames in the directory
    filenames = os.listdir(directory)

    # Define a sorting key function to extract the integer value from the filenames
    def sort_key(filename):
        # Extract the integer value from the filename
        value = int(filename[:-1])
        return value

    # Sort the filenames based on the integer value
    sorted_filenames = sorted(filenames, key=sort_key)

    return sorted_filenames

def create_psnr_ssim_graph(data, save_dir):
    for metric_name in ["psnr", "ssim"]:
        scale_factors = list(data.keys())
        metric_values = [data[scale][metric_name] for scale in scale_factors]

        # Create a new figure and axis
        fig, ax = plt.subplots()

        # Plotting the graph
        ax.plot(scale_factors, metric_values, label=metric_name)

        # Adding labels and title
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('Metric Value')
        ax.set_title(metric_name)
        ax.legend()

        # Add grid to the plot
        #ax.grid(True)

        # Add text annotations to the points
        for i, value in enumerate(metric_values):
            ax.text(scale_factors[i], value, f"{value:.2f}", ha='center', va='bottom')

        # Save the graph
        graph_path = os.path.join(save_dir, f"{metric_name}_scores_graph.png")
        plt.savefig(graph_path)
        plt.close()

        print(f"{metric_name} scores graph saved successfully.")
