import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

def process_image(model, image_path, scale):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.eval()

    image = pil_image.open(image_path).convert('RGB')

    # image_width = (image.width // scale) * scale
    # image_height = (image.height // scale) * scale
    # image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    # image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(image_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)  # Change to directory argument
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    image_paths = [os.path.join(args.image_dir, filename) for filename in os.listdir(args.image_dir) if filename.endswith(('.jpg', '.png'))]

    for image_path in image_paths:
        #print(image_path)
        process_image(model, image_path, args.scale)
