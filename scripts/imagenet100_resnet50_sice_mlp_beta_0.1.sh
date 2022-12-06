#!/bin/bash
#SBATCH --job-name=beta0
#SBATCH --time=30:00:00
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:4
#SBATCH --mem=20g

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
#module load pytorch/1.9.0-py39-cuda112 torchvision cvxpylayers
conda activate convnext
cd /scratch2/rah025

set -e
:<<!
*****************Instruction*****************
Here you can easily creat a model by selecting
an arbitray backbone model and global method.
You can fine-tune it on your own datasets by
using a pre-trained model.
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
#mpncovresnet: mpncovresnet50, mpncovresnet101
#inceptionv3
#You can also add your own network in src/network
arch=resnet50
#*********************************************

#***************global method****************
#Our code provides some global method at the end
#of network:
#GAvP (global average pooling),
#MPNCOV (matrix power normalized cov pooling),
#BCNN (bilinear pooling)
#CBP (compact bilinear pooling)
#...
#You can also add your own method in src/representation
image_representation=SICE_MLP_RES
# short description of method
description=reproduce
#*********************************************

#*******************Dataset*******************
#Choose the dataset folder
benchmark=imagenet100
datadir=dataset
dataset=$datadir/$benchmark
num_classes=100
#*********************************************

#****************Hyper-parameters*************

# Freeze the layers before a certain layer.
freeze_layer=0
# Batch size
batchsize=50
# The number of total epochs for training
epoch=100
# The inital learning rate
# dcreased by step method
#lr=1.2e-6
lr=0.01
lr_method=step
lr_params=15\ 30\ 45
# log method
# description: lr = logspace(params1, params2, #epoch)

#lr_method=log
#lr_params=-1.1\ -5.0
weight_decay=1e-4
classifier_factor=1

#append lambda value
sparsity=0.01
iterations=5
sicelr=1.0

beta=0.1
#*********************************************
echo "Start finetuning!"
modeldir=Results/deepSICE-$benchmark-$arch-$image_representation-$description-lr$lr-bs$batchsize-mlp-beta-$beta
if [ ! -d  "Results" ]; then

mkdir Results

fi
if [ ! -e $modeldir/*.pth.tar ]; then

if [ ! -d  "$modeldir" ]; then

mkdir $modeldir

fi
#cp finetune_sice_mpncovvggd16.sh $modeldir

python main_mlp.py $dataset\
               --benchmark $benchmark\
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
               --sicelr $sicelr\
               --beta $beta

else
checkpointfile=$(ls -rt $modeldir/*.pth.tar | tail -1)

python main_mlp.py $dataset\
               --benchmark $benchmark\
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
               --sicelr $sicelr\
               --beta $beta
fi
echo "Done!"
