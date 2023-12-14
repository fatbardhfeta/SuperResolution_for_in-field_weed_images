# SuperResolution for in-field weed_images

This is a repo for my thesis project "Zoom & Enhance: SuperResolution for in-field weed detection"

## Table of contents

- [Introduction](#problem-introduction)
    - [Problem definition](#problem-definition)

- [Code](#code)
    - [Prepare Data](#prepare_data)
    - [Run Filtering Functions](#run-filtering-functions)
    - [Run segmentation](#run-segmentation)
    - [Run Inference](#run-inference)
- [Models](#models)
- [Online Resources](#online-resources)

## Introduction

Agriculture is crucial to human progress, with an impending challenge as food demand is projected to increase significantly by 2050 due to global population growth. This thesis explores the integration of Unmanned Aerial Vehicles (UAVs) and Super-resolution methods in agriculture to meet these growing demands.

### Problem definition

The primary challenge addressed in this work is the efficient early-stage detection of weeds in large fields of maize using UAVs, a task made challenging by the need for high-resolution imaging at low flight altitudes. This work evaluates super-resolution techniques to enhance low-resolution UAV images taken at higher altitudes.

## Code

### Preparin Data
First, create a virtual environment using virtualenvwrapper:

```shell
source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv segmentation-env
```

Install the required dependencies inside the created environment:

```shell
cd SuperResolution_for_in-field_weed_images/inference/segmentation
pip install -r requirements.txt
```

To train and test our methods, you need to create the datasets first. We divide high-resolution images into patches and downscale them bicubicaly. An example config file is shown at 'configs/create_training_dataset.json' and 'configs/create_test_dataset.json'. To create the datasets, run the following commands:

```shell
python data/create_training_dataset.py --json_file_path /path/to/HR/images
```

```shell
python data/create_test_dataset.py --json_file_path /path/to/HR_test_dataset/images
```


### Run Filtering Functions
First, create a virtual environment using virtualenvwrapper or reuse the one from above:

```shell
source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv segmentation-env
```

Install the required dependencies inside the created environment:

```shell
cd SuperResolution_for_in-field_weed_images/inference/segmentation
pip install -r requirements.txt
```
To test the filtering methods, you need images that have been upscaled bicubically. To do this, run the `create_test_dataset.py` script with the `'type': 'downscale&upscale'` attribute. This will create datasets of images downscaled and upscaled bicubically. Make a copy of the dataset for each filtering method:

```shell
# Copy the contents to the bilateral_filter_6m directory
cp -r results_bicubical_6m/* results_non_nn_6m/bilateral_filter_6m/

# Copy the contents to the gaussian_filter_6m directory
cp -r results_bicubical_6m/* results_non_nn_6m/gaussian_filter_6m/

# Copy the contents to the median_filter_6m directory
cp -r results_bicubical_6m/* results_non_nn_6m/median_filter_6m/

# Copy the contents to the non_local_means_denoising_6m directory
cp -r results_bicubical_6m/* results_non_nn_6m/non_local_means_denoising_6m/
```

Next, update the config file in 'configs/create_filtering_configuration.json'. Then to run the filtering methods run:

```shell
python insights/run_filtering.py
```



### Run segmentation
First, create a virtual environment using virtualenvwrapper or reuse the one from above:

```shell
source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv segmentation-env
```

Install the required dependencies inside the created environment:

```shell
cd SuperResolution_for_in-field_weed_images/inference/segmentation
pip install -r requirements.txt
```

Then make sure to update the config file under `configs/segmentation_configuration.json`. You can run multiple segmentation experiments at ounce. To run the segmentations, use the following commands:

```shel
python segmentat_images.py --json_file_path segmentation_config.json

```

### Run Inference
We can run multiple inference tasks at one. To run the inference script on the segmentations produced above follow the next steps. 
First, make sure to be working on the same enviroment as above. Then update the configuartion script from here 'configs/run_inference_configuration.json'.
Then run:

```shell
python inference/segmentation/run_inference.py
```

In the config, for each inference case you have to determine:
- model_path: path of the model
- images_dir: directory with images to test
- masks_dir (optional): the directory with the masks of the images to segment. If not available, the segmentation of the real image will be used as ground truth.


## Models
More on the ML models used in this work, please refer to [this](./models/README.md) page.

## Online Resources

In this work we have finetuned many networks and created a wide array of data. Since we can not put all of them in this repo, you can find them here:

