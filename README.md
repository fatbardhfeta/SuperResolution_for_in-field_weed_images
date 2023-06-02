# SuperResolution for in-field weed_images
This is a repo for my thesis project "Zoom & Enhance: SuperResolution for in-field weed detection" 

## Table of contents

- [Introduction](#problem-introduction)
    - [Problem definition](#problem-definition)

- [Code](#code)
    - [Running Inference](#)


## Introduction

### Problem definition

## Code

### Running segmentation

First create a virtual enviroment, we use virtual enviroment wraper: 

    source /usr/local/bin/virtualenvwrapper.sh
    mkvirtualenv segmentation-env


In the enviroment you created install the requirements file:

    cd SuperResolution_for_in-field_weed_images/inference/segmentation

    pip install -r requirements.txt

To run the segmentations:

    cd SuperResolution_for_in-field_weed_images/
    python3 inference/segmentation/predict_2021_data.py

You have to change the dir where the images you want to segment are located. This can be done if you open the script 'redict_2021_data.py'. The new segmented images are under 'inference/segmentation/results/example'.


To calculate the dice score for different bicubical zoom factors for a given model and dataset run this 

    python3 inference/segmentation/evaluate_segmentation.py --model_path inference/segmentation/models/model.h5 --images_dir /home/fatbardhf/data_code/data/flight_altitude/test/images --masks_dir /home/fatbardhf/data_code/data/flight_altitude/test/masks

- model_path: path of model 
- images_dir: dir with images to test
- masks_dir( optional): the dir with the masks of the images to segment. in casethis is not aveable the segmentation of the real image will be used as ground truth

README file will be updated during development
