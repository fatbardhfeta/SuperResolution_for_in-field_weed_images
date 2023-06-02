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

README file will be updated during development
