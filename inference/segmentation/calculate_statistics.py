import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

hr_folder = "inference/segmentation/results/HR_segmentations"  # HR segmentation folder
x2_folder = "inference/segmentation/results/results_x2"  # 2x segmentation folder
x3_folder = "inference/segmentation/results/results_x3"  # 3x segmentation folder
x4_folder = "inference/segmentation/results/results_x4"  # 4x segmentation folder

def calculate_psnr_ssim(hr_path, x_path):
    hr_image = np.array(Image.open(hr_path))
    x_image = np.array(Image.open(x_path))

    psnr = peak_signal_noise_ratio(hr_image, x_image)
    ssim = structural_similarity(hr_image, x_image)

    return psnr, ssim

def main():
    # Get the list of HR segmentation images
    hr_images = sorted(os.listdir(hr_folder))

    statistics = []  # List to store statistics for all comparisons

    for hr_image in hr_images:
        hr_path = os.path.join(hr_folder, hr_image)
        x2_image = f"model2021_2x_{hr_image.split('_')[2]}_pred.png"
        x3_image = f"model2021_3x_{hr_image.split('_')[2]}_pred.png"
        x4_image = f"model2021_4x_{hr_image.split('_')[2]}_pred.png"
        x2_path = os.path.join(x2_folder, x2_image)
        x3_path = os.path.join(x3_folder.replace("x3", "2x"), x3_image)
        x4_path = os.path.join(x4_folder, x4_image)

        psnr_x2, ssim_x2 = calculate_psnr_ssim(hr_path, x2_path)
        psnr_x3, ssim_x3 = calculate_psnr_ssim(hr_path, x3_path)
        psnr_x4, ssim_x4 = calculate_psnr_ssim(hr_path, x4_path)

        statistics.append({
            "Image": hr_image,
            "2x PSNR": psnr_x2,
            "2x SSIM": ssim_x2,
            "3x PSNR": psnr_x3,
            "3x SSIM": ssim_x3,
            "4x PSNR": psnr_x4,
            "4x SSIM": ssim_x4
        })

    # Print statistics table
    print("Comparison Statistics:")
    print("-------------------------------------------------------")
    print("{:<20s} {:<10s} {:<10s} {:<10s} {:<10s}".format(
        "Image", "2x PSNR", "2x SSIM", "3x PSNR", "3x SSIM"))
    print("-------------------------------------------------------")
    for stats in statistics:
        print("{:<20s} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            stats["Image"], stats["2x PSNR"], stats["2x SSIM"], stats["3x PSNR"], stats["3x SSIM"]))
    print("-------------------------------------------------------")

    # Generate graphs
    image_names = [stats["Image"] for stats in statistics]
    x2_psnr_values = [stats["2x PSNR"] for stats in statistics]
    x3_psnr_values = [stats["3x PSNR"] for stats in statistics]
    x4_psnr_values = [stats["4x PSNR"] for stats in statistics]
    x2_ssim_values = [stats["2x SSIM"] for stats in statistics]
    x3_ssim_values = [stats["3x SSIM"] for stats in statistics]
    x4_ssim_values = [stats["4x SSIM"] for stats in statistics]

    # Plot PSNR graph
    plt.figure(figsize=(10, 5))
    plt.plot(image_names, x2_psnr_values, label="2x")
    plt.plot(image_names, x3_psnr_values, label="3x")
    plt.plot(image_names, x4_psnr_values, label="4x")
    plt.xlabel("Image")
    plt.ylabel("PSNR")
    plt.title("PSNR Comparison")
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()

    # Plot SSIM graph
    plt.figure(figsize=(10, 5))
    plt.plot(image_names, x2_ssim_values, label="2x")
    plt.plot(image_names, x3_ssim_values, label="3x")
    plt.plot(image_names, x4_ssim_values, label="4x")
    plt.xlabel("Image")
    plt.ylabel("SSIM")
    plt.title("SSIM Comparison")
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()

if __name__ == "__main__":
    main()
