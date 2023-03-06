# Learning Partial Correlation based Deep Visual Representation for Visual Classification, To be appeared in CVPR 2023

## Introduction
The official repository for paper "Learning Partial Correlation based Deep Visual Representation for Image Classification" To appear in 2023 The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR).

![iSICE](framework.png)

Our partial correlation based deep visual representation method iSICE is implemented using PyTorch as a meta-layer. iSICE is fully differentiable and optimisable with torch autograd package. Our codebase designed in a flexible manner so that any change in the pipeline can be done more convaniently. Specifically, we considered separating backone networks, global image representation and classification steps so that each step can be changed without affecting others. We find that such desgin is helpful to conduct hassle-free experiments and reduce issues during future extensions.

The following backbones are supported at this moment: AlexNet, VGG-11, VGG-11_BN, VGG-13, VGG-13_BN, VGG16, VGG16_BN, VGG19, VGG19_BN, ResNet-18, ResNet-34, resnet50, ResNet-101, ResNet-152, Inception-V3

Besides iSICE, we have added the following global representation methods in the repository: Global Average Pooling, iSQRT-COV Pooling, Bilinear Pooling (B-CNN), Compact Bilinear Pooling (CBP), etc.

## How to use the code

1. Install anaconda and create a conda environment with the following packages (some packages maybe installed with pip).
    1. torch 1.9 (install with CuDNN support)
    2. torchvision 0.13.0
    3. matplotlib
    4. scipy
    5. numpy

2. All of the datasets used for experiments are publicly available online. Please download them and prepare the dataset as follows.
```
    .
    ├── train
    │   ├── class1
    │   │   ├── class1_001.jpg
    │   │   ├── class1_002.jpg
    |   |   └── ...
    │   ├── class2
    │   ├── class3
    │   ├── ...
    │   ├── ...
    │   └── classN
    └── val
        ├── class1
        │   ├── class1_001.jpg
        │   ├── class1_002.jpg
        |   └── ...
        ├── class2
        ├── class3
        ├── ...
        ├── ...
        └── classN
```

3. for finetuning our iSICE model modify the fields in fine-tune.sh script
