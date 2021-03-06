#!/bin/bash
set -e
set -x

# Point to pre-trained models.
export MODEL_DIR="pretrained_models"

# Human mask experiments.
#export DATA_DIR="ANN_recognition_data_human_masks"
#export OUTPUT_DIR="ANN_recognition_outputs_human_masks"

# ANN mask experiments.
export DATA_DIR="ANN_recognition_data_ANN_masks"
export OUTPUT_DIR="ANN_recognition_outputs_ANN_masks"

mkdir -p $OUTPUT_DIR

# Baseline CIFAR-100 AlexNet:
python ../baseline_cnns/get_cifar_confidences.py \
  --arch alexnet \
  --data $DATA_DIR \
  --confidences_out $OUTPUT_DIR/CIFAR-100_baseline-cnns_AlexNet.txt \
  --resume $MODEL_DIR/baseline_cnns/alexnet/model_best.pth.tar

# Baseline CIFAR-100 VGG-19 with BatchNorm:
python ../baseline_cnns/get_cifar_confidences.py \
  --arch vgg19_bn \
  --data $DATA_DIR \
  --confidences_out $OUTPUT_DIR/CIFAR-100_baseline-cnns_VGG-19-BN.txt \
  --resume $MODEL_DIR/baseline_cnns/vgg19_bn/model_best.pth.tar

# Baseline CIFAR-100 ResNet-110:
python ../baseline_cnns/get_cifar_confidences.py \
  --data $DATA_DIR \
  --confidences_out $OUTPUT_DIR/CIFAR-100_baseline-cnns_ResNet-110.txt \
  --arch resnet \
  --resume $MODEL_DIR/baseline_cnns/resnet-110/model_best.pth.tar

# Baseline ImageNet AlexNet:
python ../baseline_cnns/get_imagenet_confidences.py \
  --arch alexnet \
  --confidences_out $OUTPUT_DIR/ImageNet_baseline-cnns_AlexNet.txt \
  --data $DATA_DIR

# Baseline ImageNet VGG-16 with BatchNorm:
python ../baseline_cnns/get_imagenet_confidences.py \
  --arch vgg16_bn \
  --confidences_out $OUTPUT_DIR/ImageNet_baseline-cnns_VGG-16-BN.txt \
  --data $DATA_DIR

# Baseline ImageNet ResNet-101:
python ../baseline_cnns/get_imagenet_confidences.py \
  --data $DATA_DIR \
  --confidences_out $OUTPUT_DIR/ImageNet_baseline-cnns_ResNet-101.txt \
  --arch resnet101

# Baseline ImageNet EfficientNet-B0:
python ../baseline_cnns/get_imagenet_confidences.py \
  --arch efficientnet \
  --confidences_out $OUTPUT_DIR/ImageNet_baseline-cnns_EfficientNet-B0.txt \
  --data $DATA_DIR

# Baseline Places365 AlexNet:
python ../baseline_cnns/get_places_confidences.py \
  --arch alexnet \
  --data $DATA_DIR \
  --confidences_out $OUTPUT_DIR/Places365_baseline-cnns_AlexNet.txt \
  --resume $MODEL_DIR/baseline_cnns/alexnet/alexnet_places365.pth.tar

# Baseline Places365 ResNet-50:
python ../baseline_cnns/get_places_confidences.py \
  --arch resnet50 \
  --data $DATA_DIR \
  --confidences_out $OUTPUT_DIR/Places365_baseline-cnns_ResNet50.txt \
  --resume $MODEL_DIR/baseline_cnns/resnet-50/resnet50_places365.pth.tar

# Baseline ImageNet ViT-small:
python ../baseline_cnns/get_imagenet_confidences.py \
  --arch vit \
  --confidences_out $OUTPUT_DIR/ImageNet_baseline-cnns_ViT-small.txt \
  --data $DATA_DIR

# CIFAR-100 ABN ResNet-110:
python ../attention-branch-network/get_cifar_confidences.py \
  --data $DATA_DIR \
  --confidences_out $OUTPUT_DIR/CIFAR-100_attention-branch-network_ResNet-110.txt \
  --arch resnet \
  --model $MODEL_DIR/attention-branch-network/pretrained-cifar100-resnet110/model_best.pth.tar

# CIFAR-100 ABN DenseNet-BC:
python ../attention-branch-network/get_cifar_confidences.py \
  --data $DATA_DIR \
  --confidences_out $OUTPUT_DIR/CIFAR-100_attention-branch-network_DenseNet-BC.txt \
  --arch densenet --depth 100 \
  --model $MODEL_DIR/attention-branch-network/pretrained-cifar100-densenet-bc/model_best.pth.tar

# ImageNet ABN ResNet-101:
python ../attention-branch-network/get_imagenet_confidences.py \
  --data $DATA_DIR \
  --confidences_out $OUTPUT_DIR/ImageNet_attention-branch-network_ResNet-101.txt \
  --arch resnet101 \
  --model $MODEL_DIR/attention-branch-network/pretrained-imagenet2012-resnet101/model_best.pth.tar

# CIFAR-100 LTPA VGG:
python ../learn-to-pay-attention/get_confidences.py \
  --data $DATA_DIR \
  --confidences_out $OUTPUT_DIR/CIFAR-100_learn-to-pay-attention_VGG.txt \
  --model $MODEL_DIR/learn-to-pay-attention/pretrained-before/net.pth --normalize_attn
