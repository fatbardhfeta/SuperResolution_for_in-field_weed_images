import json
import argparse
import csv
from datetime import date

from tools.evaluation_functions import * 

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Segmentation evaluation script")
    parser.add_argument("--json_file_path", type=str, default="/home/fatbardhf/SuperResolution_for_in-field_weed_images/configs/run_inference_configuration.json", help="Path to json config file")
    args = parser.parse_args()

    # Read configuration from JSON file
    with open(args.json_file_path, mode='r') as file:
        config = json.load(file)

    for case in config["inference_cases"]:
        original_images_dir = case["original_images_path"]
        original_images =[os.path.join(original_images_dir, filename) for filename in sorted(os.listdir(original_images_dir))
                   if filename.endswith(".jpg") or filename.endswith(".png")]
        
        upscaled_dir = case["upscaled_images_path"]
        #upscaled_images_dirs = [os.path.join(upscaled_dir, directory) for directory in case["scaling_factors"]]
        upscaled_images_dirs = [os.path.join(upscaled_dir, directory) for directory in sort_filenames_by_integer_value(upscaled_dir) if os.path.isdir(os.path.join(upscaled_dir, directory))]
        
        pred = {'original' : load_masks( case["masks_path"])}

        # Load masks
        for index in case["scaling_factors"]:
            upscaled_image_dir = os.path.join(upscaled_dir, index, "segmentations")
            pred[index] = load_masks(upscaled_image_dir)

        print("Calculating dice scores:")
        print(pred.keys)
        dice_scores = calculate_dice_scores(pred)

        # Calculate psner-ssim
        psnr_ssim = {}
        # Load images
        images = {'original' : original_images}
        for downscaled_image_folder, index in zip(upscaled_images_dirs, case["scaling_factors"]):
            downscaled_image_folder = os.path.join(downscaled_image_folder, "images")
            images[index] = [os.path.join(downscaled_image_folder, filename) for filename in sorted(os.listdir(downscaled_image_folder))
                    if filename.endswith(".jpg") or filename.endswith(".png")]
        
        print("Calculating psnr and ssim:")
        psnr_ssim = calculate_images_psnr_ssim(images)

        #Save files as csv
        experiments_path =  os.path.join(config["reports_dir"], case["inference_case_name"])
        # Check if the folder exists, create it if it doesn't
        if not os.path.exists(experiments_path):
            os.makedirs(experiments_path)

        dice_graph_scores = {}
        original_masks_sizes = inspect_masks_size(pred["original"], key='original')
        for key in dice_scores.keys():
            # Combining arrays using zip
            pred_class_sizes = inspect_masks_size(pred[key], key)
            data = zip( original_masks_sizes[:,0], original_masks_sizes[:,1], original_masks_sizes[:, 2], pred_class_sizes[:, 0], pred_class_sizes[:,1], pred_class_sizes[:,2], dice_scores[key][:,0], dice_scores[key][:,1], psnr_ssim[key]["psnr"], psnr_ssim[key]["ssim"], sorted(os.listdir(original_images_dir)))

            filename = os.path.join( experiments_path,f"{key}_statistics.csv") 
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile) 
                writer.writerow([ 'Mask_class_0', 'Mask_class_1', 'Mask_class_2', 'Pred_class_0', 'Pred_class_1', 'Pred_class_2', 'Dice_class1', 'Dice_class_2', 'psnr', 'ssim', 'patch name'])  # Write header row
                writer.writerows(data)  # Write data rows

            dice_graph_scores[key]= np.nanmean(dice_scores[key], axis=0)

        save_dice_scores_graph(dice_graph_scores, experiments_path)

main()