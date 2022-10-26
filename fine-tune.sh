#!/bin/bash

# load anaconda and activate the environment
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate your_environment_name

# please provide the path of source code
cd /path/to/your/files

set -e
:<<!
*****************Instruction*****************
***Source code for AAAI Submission ID 7654***
*********************************************
Here you can easily creat a model by selecting
an arbitray backbone model and global method.
You can fine-tune it on fine-grained datasets by
using a pre-trained model listed below.
Modify the following settings as you wish !
*********************************************
!

#***************Backbone model****************
#Our code provides some mainstream architectures:
#alexnet
#vgg family:vgg11, vgg11_bn, vgg13, vgg13_bn,
#           vgg16, vgg16_bn, vgg19_bn, vgg19
#resnet family: resnet18, resnet34, resnet50,
#               resnet101, resnet152
#inceptionv3
#convnext_tiny
#swin_t
arch=convnext_tiny
#*********************************************

#***************global method****************
#Our code provides some global method at the end
#of network:
#GAvP (global average pooling),
#COV (matrix power normalized cov pooling),
#INVCOV (inverse covariance pooling)
#SICE (iterative sparse inverse covarinace pooling)
image_representation=SICE
# short description of method
description=reproduce
#*********************************************

#*******************Dataset*******************
#Choose the dataset name and folder (download them online)
#We give example of using MIT dataset
benchmark=MIT
datadir=dataset
dataset=$datadir/$benchmark
num_classes=67
#*********************************************

#****************Hyper-parameters*************
# Batch size
batchsize=50
# The number of total epochs for training
epoch=100
# The inital learning rate
# dcreased by step method
lr=5e-5
lr_method=step
lr_params=15\ 30
# log method
# description: lr = logspace(params1, params2, #epoch)

#lr_method=log
#lr_params=-1.1\ -5.0
weight_decay=0.05
classifier_factor=1

#SICE hyper-parameters used in the paper
sparsity=0.01
iterations=5
sicelr=1.0
#*********************************************
echo "Start finetuning!"
#default output path (change it according to your need)
modeldir=Results/SICE-$benchmark-$arch-$image_representation-$description-lr$lr-bs$batchsize-sparsity$sparsity-iterations$iterations-sicelr-$sicelr
if [ ! -d  "Results" ]; then

mkdir Results

fi
if [ ! -e $modeldir/*.pth.tar ]; then

if [ ! -d  "$modeldir" ]; then

mkdir $modeldir

fi
#cp finetune_sice_mpncovvggd16.sh $modeldir

python main.py $dataset\
               --benchmark $benchmark\
               --pretrained\
               -a $arch\
               -p 10\
               --epochs $epoch\
               --lr $lr\
               --lr-method $lr_method\
               --lr-params $lr_params\
              --weight-decay $weight_decay\
               -j 8\
               -b $batchsize\
               --num-classes $num_classes\
               --representation $image_representation\
               --freezed-layer $freeze_layer\
               --classifier-factor $classifier_factor\
               --benchmark $benchmark\
               --modeldir $modeldir\
               --sparsity $sparsity\
               --iterations $iterations\
               --sicelr $sicelr

else
checkpointfile=$(ls -rt $modeldir/*.pth.tar | tail -1)

python main.py $dataset\
               --benchmark $benchmark\
               --pretrained\
               -a $arch\
               -p 10\
               --epochs $epoch\
               --lr $lr\
               --lr-method $lr_method\
               --lr-params $lr_params\
               --weight-decay $weight_decay\
               -j 8\
               -b $batchsize\
               --num-classes $num_classes\
               --representation $image_representation\
               --freezed-layer $freeze_layer\
               --modeldir $modeldir\
               --classifier-factor $classifier_factor\
               --benchmark $benchmark\
               --resume $checkpointfile\
               --sparsity $sparsity\
               --iterations $iterations\
               --sicelr $sicelr
fi
echo "Done!"
