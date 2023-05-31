import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import numpy as np

from tabulate import tabulate
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Segmentation files
hr_seg_folder = "inference/segmentation/results/HR_segmentations"  # HR segmentation folder
x2_seg_folder = "inference/segmentation/results/results_2x"  # 2x segmentation folder
x3_seg_folder = "inference/segmentation/results/results_3x"  # 3x segmentation folder
x4_seg_folder = "inference/segmentation/results/results_4x"  # 4x segmentation folder

# Downsampled files
hr_folder = "/home/fatbardhf/data_code/data/A_20210707/png_lenscor"  # HR folder
x2_folder = "/home/fatbardhf/bicubical_A_20210707/2x"  # 2x bicubivaly downsampled
x3_folder = "/home/fatbardhf/bicubical_A_20210707/3x"  # 3x bicubivaly downsampled
x4_folder = "/home/fatbardhf/bicubical_A_20210707/4x"  # 4x bicubivaly downsampled




def substitute_colors(image):
    '''
    Substitutes specific RGB values in an image with new values.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The image with substituted colors.
    '''
    substitution_dict = {
    (199, 199, 199): (1, 0, 0),
    (31, 119, 180): (0, 1, 0),
    (255, 127, 14): (0, 0, 1)
}
    result = np.copy(image)
    for original_color, new_color in substitution_dict.items():
        mask = np.all(result == original_color, axis=-1)
        result[mask] = new_color
    return result

def compare_segmentations(hr_path, x2_path, x3_path, x4_path):
    '''' Compare the images pixel wise'''
    hr_image = Image.open(hr_path)
    x2_image = Image.open(x2_path)
    x3_image = Image.open(x3_path)
    x4_image = Image.open(x4_path)

    hr_pixels = substitute_colors(np.array(hr_image))
    x2_pixels = substitute_colors(np.array(x2_image))
    x3_pixels = substitute_colors(np.array(x3_image))
    x4_pixels = substitute_colors(np.array(x4_image))

    class_errors_hr_x2 = np.sum(hr_pixels != x2_pixels, axis=(0, 1))
    class_errors_hr_x3 = np.sum(hr_pixels != x3_pixels, axis=(0, 1))
    class_errors_hr_x4 = np.sum(hr_pixels != x4_pixels, axis=(0, 1))

    total_pixels = hr_pixels.shape[0] * hr_pixels.shape[1]

    return class_errors_hr_x2, class_errors_hr_x3, class_errors_hr_x4, total_pixels


def calculate_psnr_ssim(original_path, downsampled_path):
    original_image = np.array(Image.open(original_path))
    downsampled_image = np.array(Image.open(downsampled_path))
    psnr = peak_signal_noise_ratio(original_image, downsampled_image)
    ssim = structural_similarity(original_image, downsampled_image, channel_axis = 2, multichannel=True)

    return psnr, ssim

def main():
 # Get the list of HR segmentation images
    hr_seg_images = sorted(os.listdir(hr_seg_folder))

    class_errors_hr_x2_total = np.zeros(3)
    class_errors_hr_x3_total = np.zeros(3)
    class_errors_hr_x4_total = np.zeros(3)
    total_pixels = 0

    # Calculate segmentation errors
    for hr_seg_image in tqdm(hr_seg_images, desc="Calculating error of segmentations: ", unit="image"):
        hr_path = os.path.join(hr_seg_folder, hr_seg_image)
        temp = hr_seg_image.split('_')[2:]
        hr_image_name = f"{temp[0]}_{temp[1]}_{temp[2]}_{temp[3]}"  # Extract the image name from the HR segmentation file name
        x2_image = f"model2021_2x_img_{hr_image_name}_pred.png"
        x3_image = f"model2021_3x_img_{hr_image_name}_pred.png"
        x4_image = f"model2021_4x_img_{hr_image_name}_pred.png"
        x2_path = os.path.join(x2_seg_folder, x2_image)
        x3_path = os.path.join(x3_seg_folder, x3_image)
        x4_path = os.path.join(x4_seg_folder, x4_image)

        class_errors_hr_x2, class_errors_hr_x3, class_errors_hr_x4, total_pixels_img = compare_segmentations(hr_path, x2_path, x3_path, x4_path)

        class_errors_hr_x2_total += class_errors_hr_x2
        class_errors_hr_x3_total += class_errors_hr_x3
        class_errors_hr_x4_total += class_errors_hr_x4
        total_pixels += total_pixels_img

    average_errors_hr_x2 = class_errors_hr_x2_total / total_pixels
    average_errors_hr_x3 = class_errors_hr_x3_total / total_pixels
    average_errors_hr_x4 = class_errors_hr_x4_total / total_pixels

    # Create the table data
    table_data = [
        ["2x"] + [f"{error * 100:.4f}%" for error in average_errors_hr_x2],
        ["3x"] + [f"{error * 100:.4f}%" for error in average_errors_hr_x3],
        ["4x"] + [f"{error * 100:.4f}%" for error in average_errors_hr_x4],
    ]

    # Define the table headers
    headers = ["Scale"] + [f"Class {i+1}" for i in range(3)]

    # Print the table
    table1 = tabulate(table_data, headers, tablefmt="grid")
    print("Average Class Errors:")
    print(table1)


    
    #calculate recontruction ssim and psnr for downscaled images
    # Get the list of HR segmentation images
    hr_images = sorted(os.listdir(hr_folder))

    psnr_x2 = 0
    ssim_x2 = 0
    psnr_x3 = 0
    ssim_x3 = 0
    psnr_x4 = 0
    ssim_x4 = 0
    
    for hr_image in tqdm(hr_images, desc="Calculating ssim and psnr of downsampled images: ", unit="image"):
        hr_path = os.path.join(hr_folder, hr_image)
        temp = hr_image.split('_')
        hr_image_name = f"{temp[1]}_{temp[2]}_{temp[3]}_{temp[4]}"# Extract the image name from the HR file name
        x2_image = f"2x_img_{hr_image_name}"
        x3_image = f"3x_img_{hr_image_name}"
        x4_image = f"4x_img_{hr_image_name}"
        x2_path = os.path.join(x2_folder, x2_image)
        x3_path = os.path.join(x3_folder, x3_image)
        x4_path = os.path.join(x4_folder, x4_image)

        t_psnr_x2, t_ssim_x2= calculate_psnr_ssim(hr_path, x2_path)
        t_psnr_x3, t_ssim_x3= calculate_psnr_ssim(hr_path, x3_path)
        t_psnr_x4, t_ssim_x4= calculate_psnr_ssim(hr_path, x4_path)

        psnr_x2 += t_psnr_x2
        ssim_x2 += t_ssim_x2
        psnr_x3 += t_psnr_x3
        ssim_x3 += t_ssim_x3
        psnr_x4 += t_psnr_x4
        ssim_x4 += t_ssim_x4


    psnr_x2 = psnr_x2/len(hr_images)
    ssim_x2 = ssim_x2/len(hr_images)
    psnr_x3 = psnr_x3/len(hr_images)
    ssim_x3 = ssim_x3/len(hr_images)
    psnr_x4 = psnr_x4/len(hr_images)
    ssim_x4 = ssim_x4/len(hr_images)


    # Calculate average PSNR and SSIM values
    psnr_x2 = psnr_x2 / len(hr_images)
    ssim_x2 = ssim_x2 / len(hr_images)
    psnr_x3 = psnr_x3 / len(hr_images)
    ssim_x3 = ssim_x3 / len(hr_images)
    psnr_x4 = psnr_x4 / len(hr_images)
    ssim_x4 = ssim_x4 / len(hr_images)

    # Create the table data
    table_data = [
        ["2x", f"{psnr_x2:.4f}", f"{ssim_x2:.4f}"],
        ["3x", f"{psnr_x3:.4f}", f"{ssim_x3:.4f}"],
        ["4x", f"{psnr_x4:.4f}", f"{ssim_x4:.4f}"],
    ]

    # Define the table headers
    headers = ["Scale", "Average PSNR", "Average SSIM"]

    # Print the table
    table2 = tabulate(table_data, headers, tablefmt="grid")
    print(table2)

    # Save tables in file
     # Save the table output to a file
    output_file = "inference/segmentation/statistics_for_bicubical_segmentations.txt"
    with open(output_file, "w") as file:
        file.write("Average Class Errors:\n")
        file.write(table1)

        file.write("\nPSNR and SSIM:\n")
        file.write(table2)



if __name__ == "__main__":
    main()
