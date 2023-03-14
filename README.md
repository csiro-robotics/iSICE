## Partial Correlation based Deep Visual Representation<br><sub>Official implementation with PyTorch</sub>

### ![Paper](#) | ![Project Website](#) 
![iSICE](isice.png)
This repository contains the model definitions, training/evaluation code and pre-trained model weights for our paper exploring partial correlation based deep SPD visual representation. More information are available on our [project website](#).

> <b>Learning Partial Correlation based Deep Visual Representation for Image Classification</b> <br>
> [Saimunur Rahman](#), [Piotr Koniusz](http://users.cecs.anu.edu.au/~koniusz), [Lei Wang](https://sites.google.com/view/lei-hs-wang), [Luping Zhou](https://www.sydney.edu.au/engineering/about/our-people/academic-staff/luping-zhou.html), [Peyman Moghadam](https://people.csiro.au/m/p/peyman-moghadam), [Changming Sun](https://vision-cdc.csiro.au/changming.sun)<br>
> CSIRO Data61, University of Wollongong, University of Sydney

Visual representation based on covariance matrix has demonstrates its efficacy for image classification by characterising the pairwise correlation of different channels in convolutional feature maps. However, pairwise correlation will become misleading once there is another channel correlating with both channels of interest, resulting in the "confounding" effect. For this case, "partial correlation" which removes the confounding effect shall be estimated instead. Nevertheless, reliably estimating partial correlation requires to solve a symmetric positive definite matrix optimisation, known as sparse inverse covariance estimation (SICE). How to incorporate this process into CNN remains an open issue. In this work, we formulate SICE as a novel structured layer of CNN. To ensure the CNN still be end-to-end trainable, we develop an iterative method based on Newton-Schulz iteration to solve the above matrix optimisation during forward and backward propagation steps. Our work not only obtains a partial correlation based deep visual representation but also mitigates the small sample problem frequently encountered by covariance matrix estimation in CNN. Computationally, our model can be effectively trained with GPU and works well with a large number of channels in advanced CNN models. Experimental results confirm the efficacy of the proposed deep visual representation and its superior classification performance to that of its covariance matrix based counterparts.

This repository contains:

- A simple implementation of our method with PyTorch
- A script useful for training/evaluating our method on various datasets
- Pre-trained model weights on several datasets

## Introduction
The official repository for paper "Learning Partial Correlation based Deep Visual Representation for Image Classification" To appear in 2023 The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR).



Our partial correlation based deep visual representation method iSICE is implemented using PyTorch as a meta-layer. iSICE is fully differentiable and optimisable with torch autograd package. Our codebase designed in a flexible manner so that any change in the pipeline can be done more convaniently. Specifically, we considered separating backone networks, global image representation and classification steps so that each step can be changed without affecting others. We find that such desgin is helpful to conduct hassle-free experiments and reduce issues during future extensions.

The following backbones are supported at this moment: AlexNet, VGG-11, VGG-11_BN, VGG-13, VGG-13_BN, VGG16, VGG16_BN, VGG19, VGG19_BN, ResNet-18, ResNet-34, resnet50, ResNet-101, ResNet-152, Inception-V3

Besides iSICE, we have added the following global representation methods in the repository: Global Average Pooling, iSQRT-COV Pooling, Bilinear Pooling (B-CNN), Compact Bilinear Pooling (CBP), etc.

## How to use the code

### Requirements
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

3. for finetuning our iSICE model modify the fields in train_iSICE_model.sh script
