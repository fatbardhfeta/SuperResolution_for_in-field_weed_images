import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import numpy as np

from tabulate import tabulate
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from calculate_statistics import calculate_psnr_ssim


img1 = "/home/fatbardhf/data_code/data/A_20210707/png_lenscor/img_20210707_A1_0_032.png"
img2 = "/home/fatbardhf/bicubical_A_20210707/4x/4x_img_20210707_A1_0_032.png"

print(calculate_psnr_ssim(img1, img2))

