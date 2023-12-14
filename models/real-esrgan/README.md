This repository is adopted from the ["Real-ESRGAN"](https://github.com/xinntao/Real-ESRGAN) repository. The original paper can be found [here](https://arxiv.org/abs/2107.10833), titled "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"

## Environment
Before running the project, it's important to set up the necessary environment. We use virtual enviroment wraper but you can use your prefered tool:

1. **Install virtualenv and virtualenvwrapper:**


   If you haven't already installed these tools, you can do so using pip:
     ```
     pip install virtualenv virtualenvwrapper
     ```

2. **Source virtualenvwrapper.sh:**


   To use virtual environments effectively, you need to source `virtualenvwrapper.sh`. Run the following command:
     ```
     source /usr/local/bin/virtualenvwrapper.sh
     ```

3. **Create a virtual environment:**


    Now, you can create a dedicated virtual environment for your SRCNN project. Choose a name for your environment (e.g., `srcnn-env`) and create it:
     ```
     mkvirtualenv realESRGAN-env
     ```

4. **Activate the virtual environment:**


    Activate your newly created virtual environment to isolate your project's dependencies:
     ```
     workon realESRGAN-env
     ```

## Requirements

To ensure your environment has all the necessary dependencies, refer to the `requirements.txt` file in the project repository. Install these dependencies using your preferred package manager.
     ```
     pip install -r requirements.txt
     ```


## Prepare Dataset

To prepare the datsetes fot this model refer to [this](https://github.com/fatbardhfeta/SuperResolution_for_in-field_weed_images/tree/main?tab=readme-ov-file#preparin-data) section.

## Finetuned and default weights

To download the finetuned and default model weights go to this [link](https://drive.google.com/drive/folders/14K_3Xy3RmSgiwKjt9x8F296ro_QuJo3N?usp=sharing).

## Training

To train the SRCNN model, use something similar to:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4333 realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml --launcher pytorch --auto_resume
           
```


## Test
To test the srcnn model, use a comand similar to this:
```bash
python inference_realesrgan.py  -i /home/fatbardhf/plate_downsampled/6m_baseline/2x/images -o results/fintuned/2x/ -s 2 -n RealESRGAN_x2plus --suffix "" --model_path /home/fatbardhf/thesis/Real-ESRGAN/experiments/finetune_RealESRGANx2plus_400k/models/net_g_20000.pth
```


