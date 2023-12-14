# HAT - Activating More Pixels in Image Super-Resolution Transformer
This implementation is based on the repository for [HAT model](https://github.com/XPixelGroup/HAT). [Paper Link](https://arxiv.org/abs/2205.04437) 



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
     mkvirtualenv hat-env
     ```

4. **Activate the virtual environment:**


    Activate your newly created virtual environment to isolate your project's dependencies:
     ```
     workon hat-env
     ```

## Requirements
To ensure your environment has all the necessary dependencies, refer to the `requirements.txt` file in the project repository. Install these dependencies using your preferred package manager.

```
pip install -r requirements.txt
```

### Installation
Run this comand to install the application:

```
python setup.py develop
```

## How To Test
- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  

- The pretrained models are available at
[Google Drive](https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1u2r4Lc2_EEeQqra2-w85Xg) (access code: qyrl).  
- For the finetuned weights contact fatabardh.feta@tum.de

- Then run the follwing codes (taking `HAT_SRx4_ImageNet-pretrain.pth` as an example):
```
python hat/test.py -opt options/test/HAT_SRx4_ImageNet-pretrain.yml
```
The testing results will be saved in the `./results` folder.  

- Refer to `./options/test/HAT_SRx4_ImageNet-LR.yml` for **inference** without the ground truth image.



## How To Train
- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md). ImageNet dataset can be downloaded at the [official website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).

- The training command is like
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 hat/train.py -opt options/train/train_HAT_SRx2_from_scratch.yml --launcher pytorch
```
- Note that the default batch size per gpu is 4, which will cost about 20G memory for each GPU.  

The training logs and weights will be saved in the `./experiments` folder.

