# Deep-Learning Enabled Rapid and Low-Cost Detection of Microplastics via Image Processing

## Introduction

This repository is a part of [*Deep-Learning Enabled Rapid and Low-Cost Detection of Microplastics in Consumer Products Following On-site Extraction and Image Processing*](https://pubs.rsc.org/en/content/articlehtml/2025/RA/D4RA07991D), **DOI:** 10.1039/D4RA07991D.

## Table of contents

1. [Getting started](#getting-started)

    - [Prerequisites](#prerequisites)
    - [How to use](#how-to-use)

2. [Repository structure](#repository-structure)

3. [Dataset structure](#dataset-structure)

	- [Classification Class](#classification-class)

## Getting started

### Prerequisites

All codes available in the repository are written in [Python 3](https://www.python.org/). Additionally, it is highly recommended to use [Jupyter Notebook](https://jupyter.org/) or [Google Colab](https://colab.research.google.com/) for the training and testing of the ML models since it allows for the execution of small batches of code instead of running the entire script.

For object detection, we used [YOLOv5](https://github.com/ultralytics/yolov5), implemented in Google Colab using the [PyTorch](https://pytorch.org/) framework.

For data analysis, we used [Numpy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/). For data visualization purposes, we employed the [Matplotlib](https://matplotlib.org/) library (particularly the Pyplot submodule).

Listed below are the versions of the various modules used during the project:

- Python - 3.10.5
- Numpy - 1.25.2
- Pandas - 2.0.3
- Matplotlib - 3.7.1
- PyTorch - 2.0.1+cu118
- YOLOv5 - (Ultralytics GitHub release)

The versions of the modules do not seem crucial, however we still suggest using the latest versions available to ensure compatibility and performance.

### How to use

1. Clone the repository.

        go to https://github.com/ultralytics/yolov5
        !git clone https://github.com/ultralytics/yolov5  # clone
        %cd yolov5
        %pip install -qr requirements.txt comet_ml  # install

        import torch
        import utils
        display = utils.notebook_init()  # checks 

2. Upload dataset. These files should be containing "train_images.zip", "val_images.zip", "test_images.zip", "train_labels.zip", "val_labels.zip", "test_labels.zip". Put them according your designated folder.

3. Upload the "custom_data_microplastic.yaml" file and put in the designated folder for yaml files.

4. Train the model. Use batch number and epochs according to your preferences.

        !python train.py --img 640 --batch 16 --epochs 100 --data custom_data_microplastic.yaml --weights yolov5s.pt --cache

5. After training, ensure to keep the "best.pt" stored in your device.

6. The "detection.ipynb" notebook enables both the detection of microplastic elements and the counting of their occurrences. All necessary instructions are provided within the notebook.


## Repository structure

### Directories:

- **datasets**: Contains the dataset required for model training. This has "train, val, test" images and labels.

- **supplementary-videos**: Contains the real-time microplastic detection videos.


### Files:

- **custom_data_microplastic.yaml**: Used in training the model.

- **detect_images.py**: This file is used to detect microplastics from still images.

	  !python detect_images.py --weights best.pt --img 640 --conf 0.5

- **detect_videos.py**: This file is used to detect microplastics from motion pictures or videos.

	  !python detect_videos.py --weights best.pt --img 640 --conf 0.5

- **detection.ipynb**: is a notebook designed for detecting microplastic elements in images and counting their instances using YOLOv5.


## Dataset structure

The datasets is a ZIP file containing the train, val, and test directories. Each of these directories includes their respective images and label ZIP files. Details are provided in the table below.

|File name    |type  |Amount |
|-------------|------|-------|
|train_images |image |1990   |
|val_images   |image |250    |
|test_images  |image |250    |
|train_labels |label |1990   |
|val_labels   |label |250    |
|test_labels  |label |250    |


### Classification Class

The labels are assigned based on the following class:

	 - 0: Microplastic

