import os
from PIL import Image

hr_folder = "inference/segmentation/results/HR_segmentations"  # HR segmentation folder
x2_folder = "inference/segmentation/results/results_2x"  # 2x segmentation folder
x3_folder = "inference/segmentation/results/results_3x"  # 3x segmentation folder
x4_folder = "inference/segmentation/results/results_4x"  # 4x segmentation folder

def compare_segmentations(hr_path, x2_path, x3_path, x4_path):
    hr_image = Image.open(hr_path)
    x2_image = Image.open(x2_path)
    x3_image = Image.open(x3_path)
    x4_image = Image.open(x4_path)

    total_pixels = hr_image.size[0] * hr_image.size[1]
    error_hr_x2 = calculate_error(hr_image, x2_image)
    error_hr_x3 = calculate_error(hr_image, x3_image)
    error_hr_x4 = calculate_error(hr_image, x4_image)

    print(f"Comparison for {os.path.basename(hr_path)}:")
    print(f"2x Error: {error_hr_x2 / total_pixels:.4f}")
    print(f"3x Error: {error_hr_x3 / total_pixels:.4f}")
    print(f"4x Error: {error_hr_x4 / total_pixels:.4f}")
    print()

def calculate_error(image1, image2):
    assert image1.size == image2.size, "Images must have the same dimensions."

    diff_pixels = 0
    for pixel1, pixel2 in zip(image1.getdata(), image2.getdata()):
        if pixel1 != pixel2:
            diff_pixels += 1

    return diff_pixels

# Get the list of HR segmentation images
hr_images = sorted(os.listdir(hr_folder))

for hr_image in hr_images:
    hr_path = os.path.join(hr_folder, hr_image)
    x2_image = f"model2021_2x_{hr_image.split('_')[2]}_pred.png"
    x3_image = f"model2021_3x_{hr_image.split('_')[2]}_pred.png"
    x4_image = f"model2021_4x_{hr_image.split('_')[2]}_pred.png"
    x2_path = os.path.join(x2_folder, x2_image)
    x3_path = os.path.join(x3_folder, x3_image)
    x4_path = os.path.join(x4_folder, x4_image)

    compare_segmentations(hr_path, x2_path, x3_path, x4_path)
