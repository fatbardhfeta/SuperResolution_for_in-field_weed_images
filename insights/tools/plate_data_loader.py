# Code to load patches, where the plant is located at the center
from typing import Union
import numpy as np
from skimage import io as skio
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as a
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch


class UAVPatchDataset(Dataset):
    def __init__(self, data_folder, remove_thresh=None, transform=None):
        self.data_folder = data_folder
        self.remove_thresh = remove_thresh
        self.transform = transform
        self.images = self.load_rgb(b_masks=False)
        # self.masks = self.load_rgb(b_masks=True)
        # self._encode_masks(self.masks)

    def __len__(self):
        return len(self.images)

    def _encode_masks(self, rgb_mask):
        """
        encodes 4D numpy array. Sorghum is encoded as 1, Weed is encoded as 2.
        """
        rgb_mask[rgb_mask == 127] = 1
        rgb_mask[rgb_mask == 255] = 2
        return


    def load_rgb(self, b_masks):
        imgs = []
        data_folder = self.data_folder / "images"
        as_gray = False
        # if b_masks:
        #     data_folder = self.data_folder / "masks"
        #     as_gray = True
        for img_str in list(sorted(data_folder.glob("*"))):
            img = skio.imread(img_str, as_gray=as_gray)
            imgs.append(img)
        imgs = np.stack(imgs)
        return imgs

    def __getitem__(self, idx):
        image = self.images[idx]
        #mask = self.masks[idx]
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
            #mask = augmentations["mask"]

        return image


def get_patches_loader(
        data_folder: Path,  # Path to the data folder. Must have "images" and "masks" as subfolder
        filter_thresh: Union[None, float],  # percentage of background pixels for removing a patch
        split: str,  # "train", "val" or "test"
        generator: torch.Generator,  # Generator to be used for reproducibility
        batch_size: int = 20,  # size of a batch for training and testing
        num_workers: int = 2,  # number of workers
        pin_memory: bool = False,  # whether to pin memory or not
):
    """
    Loads the dataset as a PyTorch dataloader object for batching
    """
    print(f"Loading split {split}...")
    if split == "train":
        transforms = a.Compose(
            [a.HorizontalFlip(),
             a.VerticalFlip(),
             a.RandomRotate90(),
             a.Transpose(),
             a.Normalize(mean=(0.67420514, 0.55078549, 0.36809974), std=(0.67420514, 0.55078549, 0.36809974)),
             ToTensorV2()])
        shuffle = True

    elif any(substring in split for substring in ['val', 'test']):
        transforms = a.Compose([
            a.Normalize(mean=(0.67420514, 0.55078549, 0.36809974), std=(0.67420514, 0.55078549, 0.36809974)),
            ToTensorV2()])
        shuffle = False
    else:
        raise ValueError(f"Wrong name of labels_csv, please use one of ['train', 'val', 'test']")
    ds = UAVPatchDataset(data_folder=data_folder, remove_thresh=filter_thresh, transform=transforms)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers,
                            generator=generator)
    print(f"Final shape: {ds.images.shape}")
    return dataloader
