# SRCNN

This repository is adopted from the ["SRCNN-pytorch"](https://github.com/yjn870/SRCNN-pytorch) repository. The original paper can be found [here](https://arxiv.org/abs/1501.00092), titled "Image Super-Resolution Using Deep Convolutional Networks."

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
     mkvirtualenv srcnn-env
     ```

4. **Activate the virtual environment:**


    Activate your newly created virtual environment to isolate your project's dependencies:
     ```
     workon srcnn-env
     ```

## Requirements

To ensure your environment has all the necessary dependencies, refer to the `requirements.txt` file in the project repository. Install these dependencies using your preferred package manager.
     ```
     pip install -r requirements.txt
     ```


## Prepare Dataset

To train and test your SRCNN model effectively, you may need to prepare a custom dataset. This process may involve data collection, preprocessing, and organization. You can use the provided `prepare.py` script to assist with this. Follow this comand style:

```bash
python prepare.py --images-dir /path/to/images --output-path /path/to/output --patch-size 33 --stride 14 --scale 2 --eval
           
```

 Ensure that your custom dataset is appropriately structured and labeled. You will need two datastes, one for training and one for validation. We have provided the validation datasets [here](https://drive.google.com/drive/folders/1q47ysl8_aF6E_XRFb9b4RjDnefursgkq?usp=sharing). The training datasets are quite large so we have provided only the dataset for the x2 scaling factor. 



## Finetuned and default weights

To download the finetuned and default model weights go to this [link](https://drive.google.com/drive/folders/14K_3Xy3RmSgiwKjt9x8F296ro_QuJo3N?usp=sharing).

## Training

To train the SRCNN model, use something similar to:

```bash
python train.py --train-file "A_20210707_x2.h5" --eval-file "eval_A_20210707_x2.h5" --outputs-dir "outputs" --scale 2 --lr 1e-4 --batch-size 256 --num-epochs 400 --num-workers 4 --seed 123 > train_2x_output.txt
           
```

To train models for the 4x or the 8x task change the dataset and scale factor above.

## Test
To test the srcnn model, use a comand similar to this:
```bash
python test.py --weights-file "/home/.../.../outputs/x2/best.pth" --image-dir "/home/f.../6m_baseline/results_srcnn_6m/upscaled_bicubicaly/finetuned_weights/2x/images/" --scale 2
```

In the example above, we have the following parameters:
- `weights-file`: Path to the model weights file.
- `image-dir`: Path to the directory containing bicubically upscaled images.
- `scale`: The scale factor by which the images were upscaled.
