from EWISSeg.predict import predict
import torch
from EWISSeg.utils import convert_labelmap_to_color
from skimage import io as skio
from pathlib import Path
from EWISSeg.dataset import get_loader
import segmentation_models_pytorch as smp
batch_size = 50
model_save_path = "inference/segmentation/models/model.pt"
device = "cuda"
data_path = Path("/home/fatbardhf/data_code/data/A_20210707/png_lenscor")	# change this folder
save_path = Path("inference/segmentation/results") / "run_2"		# change this folder

model_suffix = "model2021"
num_img = 330
grid = (22,15)
save_path.mkdir(parents=True, exist_ok=True)

def get_slices_per_image(labelmap, grid):
    '''
    returns a list with the length of images, with the labelmap_per_image as BxWxH as each item
    '''
    slc_per_image = grid[0]*grid[1]
    num_images = int(labelmap.shape[0]/slc_per_image)
    labelmaps =[]
    for i in range(num_images):
        labelmap_per_image = labelmap[i*slc_per_image:(i+1)*slc_per_image,:,:]
        labelmaps.append(labelmap_per_image)
    return labelmaps

def reshape_by_image(labelmaps, grid, img_shape):
    preds_whole = []
    for lab in labelmaps:
        lab_full = combine_labelmap_from_slices(lab, grid=grid)
        lab_full = lab_full[0:img_shape[0], 0:img_shape[1]]
        preds_whole.append(lab_full)
    return torch.stack(preds_whole, dim=0)

def combine_labelmap_from_slices(labelmap, grid = (22,15), device="cuda"):
    """
    input: torch tensor in gpu with shape NxWxH or NxCxWxH
    takes a labelmap of the shape of BxWxH and converts it to WxH
    """
    if len(labelmap.shape) == 3:
        full_ann = torch.zeros((grid[1]*labelmap.shape[-1], grid[0]*labelmap.shape[-1]), dtype=torch.int64, device=device)
        offset = (labelmap.shape[-1],labelmap.shape[-1])
        tile_size= (labelmap.shape[-1],labelmap.shape[-1])
        placement=0
        for i in range(grid[1]):
            for j in range(grid[0]):
                full_ann[offset[1]*i:min(offset[1]*i+tile_size[1], full_ann.shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], full_ann.shape[1])] = labelmap[placement]
                placement+=1
    elif len(labelmap.shape) ==4:
        labelmap = labelmap.permute(0,2,3,1)
        full_ann = torch.zeros((grid[1]*labelmap.shape[-1], grid[0]*labelmap.shape[-1], 3), dtype=torch.int64, device=device)
        offset = (labelmap.shape[-1],labelmap.shape[-1])
        tile_size= (labelmap.shape[-1],labelmap.shape[-1])
        placement=0
        for i in range(grid[1]):
            for j in range(grid[0]):
                full_ann[offset[1]*i:min(offset[1]*i+tile_size[1], full_ann.shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], full_ann.shape[1]),:] = labelmap[placement]
                placement+=1
    else:
        raise ValueError("Wrong shape")
    return full_ann

model_save_stem = model_save_path.split('/')[-1]

loaded_model = torch.load(model_save_path, map_location=torch.device("cuda"))

encoder_name = loaded_model["encoder_name"]
model = smp.Unet(
encoder_name=encoder_name,      
encoder_weights="imagenet",    
in_channels=3,                  
classes=3,                      
)

model.load_state_dict(loaded_model["model_state_dict"]) 
model = model.to(device)

img_ls = sorted(list(data_path.rglob("*")))

img_ls = [str(img_path) for img_path in img_ls if img_path.is_file()]
for img_l in img_ls:
    img = skio.imread(img_l)
    generator = torch.Generator()
    generator.manual_seed(42)
    print(img_l)

    dataloader = get_loader(img_ls=[img_l], slc_size=256, b_crop=False, filter_thresh=0, split="test", generator=generator, batch_size=batch_size, num_workers=0, pin_memory=True)

    preds = predict(model=model, test_loader=dataloader, device=device)
    preds_labelmaps = get_slices_per_image(preds, grid)
    preds = reshape_by_image(preds_labelmaps, grid, img_shape=img.shape)
    for pred, fname in zip(preds, [img_l]):
        skio.imsave(f"{str(save_path)}/{model_suffix}_{fname.split('/')[-1].split('.')[0]}_pred.png", convert_labelmap_to_color(pred.to("cpu")), check_contrast=False)
