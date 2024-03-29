# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/99_utils.ipynb.

# %% auto 0
__all__ = ['seed_all', 'convert_labelmap_to_color', 'cm2inch', 'read_txt_as_list']

# %% ../nbs/99_utils.ipynb 2
import numpy as np
import random
import os
import torch


# %% ../nbs/99_utils.ipynb 3
def seed_all(seed):
    """
    sets the initial seed for numpy and pytorch to get reproducible results.
    One still need to restart the kernel to get reproducible results, as discussed in:
    https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeding everything to seed {seed}")
    return


# %% ../nbs/99_utils.ipynb 4
def convert_labelmap_to_color(labelmap):
    labels = np.array([(199, 199, 199), (31, 119, 180), (255, 127, 14)])
    lookup_table = np.array(labels)
    result = np.zeros((*labelmap.shape,3), dtype=np.uint8)
    np.take(lookup_table, labelmap, axis=0, out=result)
    return result


# %% ../nbs/99_utils.ipynb 5
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

# %% ../nbs/99_utils.ipynb 6
def read_txt_as_list(filepath):
    with open(str(filepath), "r") as f:
        data = f.read()
    data_into_list = data.split("\n")
    data_into_list[:] = [x for x in data_into_list if x]
    return data_into_list
