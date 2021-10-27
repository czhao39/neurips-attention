# Passive Attention in Artificial Neural Networks Predicts Human Visual Selectivity

This repository is the official code for "Passive Attention in Artificial Neural Networks Predicts Human Visual Selectivity".

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Images
The images used in our paper are under `images/`. These images are a subset of [this](https://data.mendeley.com/datasets/8rj98pp6km/1) open source dataset, provided under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Maps from Human Experiments
Maps from the human experiments are under `humans-maps/`. `human-maps-read.ipynb` gives an example of reading in and visualizing these maps.

## Evaluation

Here are reference commands for acquiring class confidences for each model, assuming that input images are located under `images/` and have the directory structure specified [here](https://pytorch.org/vision/stable/datasets.html#imagefolder) (note that, since we do not use "grouth truth" classes, you can simply place all input images in a single folder, e.g. `images/all/`):

#### Baseline CIFAR-100 AlexNet
```
python3 baseline_cnns/get_cifar_confidences.py --arch alexnet --data images --resume /path/to/pretrained/alexnet/model_best.pth.tar
```

#### Baseline CIFAR-100 VGG-19 with BatchNorm
```
python3 baseline_cnns/get_cifar_confidences.py --arch vgg19_bn --data images --resume /path/to/pretrained/vgg19_bn/model_best.pth.tar
```

#### Baseline CIFAR-100 ResNet-110
```
python3 baseline_cnns/get_cifar_confidences.py --arch resnet --data images --resume /path/to/pretrained/resnet-110/model_best.pth.tar
```

#### Baseline ImageNet AlexNet
```
python3 baseline_cnns/get_imagenet_confidences.py --arch alexnet --data images
```

#### Baseline ImageNet VGG-16 with BatchNorm
```
python3 baseline_cnns/get_imagenet_confidences.py --arch vgg16_bn --data images
```

#### Baseline ImageNet ResNet-101
```
python3 baseline_cnns/get_imagenet_confidences.py --arch resnet101 --data images
```

#### Baseline ImageNet EfficientNet-B0
```
python3 baseline_cnns/get_imagenet_confidences.py --arch efficientnet --data images
```

#### Baseline ImageNet ViT-small
```
python3 baseline_cnns/get_imagenet_confidences.py --arch vit --data images
```

#### Baseline Places365 AlexNet
```
python3 baseline_cnns/get_places_confidences.py --arch alexnet --data images --resume /path/to/pretrained/alexnet/model_best.pth.tar
```

#### Baseline Places365 ResNet-50
```
python3 baseline_cnns/get_places_confidences.py --arch resnet50 --data images --resume /path/to/pretrained/resnet50/model_best.pth.tar
```

#### CIFAR-100 ABN ResNet-110
```
python3 attention-branch-network/get_cifar_confidences.py --arch resnet --data images --model /path/to/pretrained-cifar100-resnet110/model_best.pth.tar
```

#### CIFAR-100 ABN DenseNet-BC
```
python3 attention-branch-network/get_cifar_confidences.py --arch densenet --data images --model /path/to/pretrained-cifar100-densenet/model_best.pth.tar --depth 100
```

#### ImageNet ABN ResNet-101
```
python3 attention-branch-network/get_imagenet_confidences.py --arch resnet101 --data images --model /path/to/pretrained-imagenet2012-resnet101/model_best.pth.tar
```

#### CIFAR-100 LTPA VGG
```
python3 learn-to-pay-attention/get_confidences.py --data images --model /path/to/pretrained/pretrained-before/net.pth --normalize_attn
```

## Attention Maps

Here are reference commands for acquiring attention maps:

#### Passive attention for CIFAR-100 models
```
python3 baseline_cnns/get_passive_attention_unlabeled_cifar.py -o ../out/cifar_maps --method guidedbp --data images --resume /path/to/pretrained/alexnet/model_best.pth.tar --arch alexnet
```
Available methods are `guidedbp`, `guidedbpximage`, `smoothgradguidedbp`, `gradcam`, `scorecam`. Available architectures are `alexnet`, `vgg19_bn`, `resnet`.

#### Passive attention for ImageNet models
```
python3 baseline_cnns/get_passive_attention_unlabeled_imagenet.py -o ../out/imagenet_maps --method guidedbp --data images --arch alexnet
```
Available methods are `guidedbp`, `guidedbpximage`, `smoothgradguidedbp`, `gradcam`, `scorecam`, `cameras`. Available architectures are `alexnet`, `vgg16_bn`, `resnet`, `efficientnet`, `vit`.

#### Passive attention for Places365 models
```
python3 baseline_cnns/get_passive_attention_unlabeled_places.py -o ../out/places_maps --method guidedbp --data images --arch alexnet --resume /path/to/pretrained/alexnet/model_best.pth.tar
```
Available methods are `guidedbp`, `guidedbpximage`, `smoothgradguidedbp`, `gradcam`, `scorecam`, `cameras`. Available architectures are `alexnet`, `resnet50`.

#### Active attention for CIFAR-100 ABN ResNet-110
```
python3 attention-branch-network/get_attention_cifar100.py --data images -o ../out/cifar_maps --model /path/to/pretrained-cifar100-resnet110/model_best.pth.tar --arch resnet --depth 110
```

#### Active attention for CIFAR-100 ABN DenseNet-BC
```
python3 attention-branch-network/get_attention_cifar100.py --data images -o ../out/cifar_maps --model /path/to/pretrained-cifar100-densenet/model_best.pth.tar --arch densenet
```

#### Active attention for ImageNet ABN ResNet-101
```
python3 attention-branch-network/get_attention_imagenet2012.py --data images -o ../out/imagenet_maps --model /path/to/pretrained-imagenet2012-resnet101/model_best.pth.tar --arch resnet101
```

#### Active attention for LTPA
```
python3 learn-to-pay-attention/get_attention_heatmaps.py --data images -o ../out/cifar_maps --attn_mode before --model /path/to/pretrained-before/net.pth --normalize_attn
```

## Pre-trained Models

You can download pretrained models using these links (details are provided in the READMEs under `baseline_cnns/`, `attention-branch-network/`, and `learn-to-pay-attention/`):

- [Baseline CIFAR-100 CNNs](https://mycuhk-my.sharepoint.com/:f:/r/personal/1155056070_link_cuhk_edu_hk/Documents/release/pytorch-classification/checkpoints/cifar100?csf=1&web=1&e=e4s1fa)
- [Baseline Places365 CNNs](https://github.com/CSAILVision/places365#pre-trained-cnn-models-on-places365-standard)
- Attention Branch Network
    - [CIFAR-100 Resnet-110](https://drive.google.com/open?id=1Wp7_tIXjq24KSI2VaL9V2N8NRlASLETD)
    - [CIFAR-100 DenseNet-BC (L=100, k=12)](https://drive.google.com/drive/folders/17ILqWvDJzFFZ603CpeoGaYrt6mhUF-B5)
    - [ImageNet ResNet-101](https://drive.google.com/drive/folders/1B5jBHTfskKAgNpsFm9iADn1lskn2UWyk)
- [Learn To Pay Attention](https://drive.google.com/drive/folders/1cp2Rp0FU6feaH74JFO_V9yVlijDJEFrZ)

## ANN Recognition Experiments

[`section_6`](section_6/) includes the ANN recognition experiments corresponding to Section 6 in the paper.
