import os
from PIL import Image
import numpy as np

hr_folder = "inference/segmentation/results/HR_segmentations"  # HR segmentation folder
x2_folder = "inference/segmentation/results/results_2x"  # 2x segmentation folder
x3_folder = "inference/segmentation/results/results_3x"  # 3x segmentation folder
x4_folder = "inference/segmentation/results/results_4x"  # 4x segmentation folder

def compare_segmentations(hr_path, x2_path, x3_path, x4_path):
    hr_image = Image.open(hr_path)
    x2_image = Image.open(x2_path)
    x3_image = Image.open(x3_path)
    x4_image = Image.open(x4_path)

    hr_pixels = np.array(hr_image)
    x2_pixels = np.array(x2_image)
    x3_pixels = np.array(x3_image)
    x4_pixels = np.array(x4_image)

    class_errors_hr_x2 = np.sum(hr_pixels != x2_pixels)
    class_errors_hr_x3 = np.sum(hr_pixels != x3_pixels)
    class_errors_hr_x4 = np.sum(hr_pixels != x4_pixels)

    return class_errors_hr_x2, class_errors_hr_x3, class_errors_hr_x4, hr_pixels.size

def main():
    # Get the list of HR segmentation images
    hr_images = sorted(os.listdir(hr_folder))

    class_errors_hr_x2_total = 0
    class_errors_hr_x3_total = 0
    class_errors_hr_x4_total = 0
    total_pixels = 0

    for hr_image in hr_images:
        hr_path = os.path.join(hr_folder, hr_image)
        temp = hr_image.split('_')[2:]
        hr_image_name = f"{temp[0]}_{temp[1]}_{temp[2]}_{temp[3]}"# Extract the image name from the HR segmentation file name
        x2_image = f"model2021_2x_img_{hr_image_name}_pred.png"
        x3_image = f"model2021_3x_img_{hr_image_name}_pred.png"
        x4_image = f"model2021_4x_img_{hr_image_name}_pred.png"
        x2_path = os.path.join(x2_folder, x2_image)
        x3_path = os.path.join(x3_folder, x3_image)
        x4_path = os.path.join(x4_folder, x4_image)

        class_errors_hr_x2, class_errors_hr_x3, class_errors_hr_x4, total_pixels_img = compare_segmentations(hr_path, x2_path, x3_path, x4_path)

        class_errors_hr_x2_total += class_errors_hr_x2
        class_errors_hr_x3_total += class_errors_hr_x3
        class_errors_hr_x4_total += class_errors_hr_x4
        total_pixels += total_pixels_img

    average_errors_hr_x2 = class_errors_hr_x2_total / total_pixels
    average_errors_hr_x3 = class_errors_hr_x3_total / total_pixels
    average_errors_hr_x4 = class_errors_hr_x4_total / total_pixels

    print("Average Class Errors:")
    print(f"2x Class Errors: {average_errors_hr_x2 * 100:.4f}%")
    print(f"3x Class Errors: {average_errors_hr_x3 * 100:.4f}%")
    print(f"4x Class Errors: {average_errors_hr_x4 * 100:.4f}%")

if __name__ == "__main__":
    main()
