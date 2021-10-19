#!/bin/bash

set -e

for ARCH in "alexnet" "vgg19_bn" "resnet"
do
    for IMAGEDIR in "../all_images/new_selected" "../all_images/new_selected_2" "../all_images/new_selected_3" "../all_images/new_selected_4" "../all_images/next_25_selected" "../all_images/next_15_selected"
    do
        echo "guidedbp"
        python3 get_passive_attention_unlabeled_cifar.py -o ../out/new_maps -m guidedbp --data $IMAGEDIR --arch $ARCH --resume pretrained-cifar-$ARCH/model_best.pth.tar

        echo "guidedbpximage"
        python3 get_passive_attention_unlabeled_cifar.py -o ../out/new_maps -m guidedbpximage --data $IMAGEDIR --arch $ARCH --resume pretrained-cifar-$ARCH/model_best.pth.tar

        echo "smoothgradguidedbp"
        python3 get_passive_attention_unlabeled_cifar.py -o ../out/new_maps -m smoothgradguidedbp --data $IMAGEDIR --arch $ARCH --resume pretrained-cifar-$ARCH/model_best.pth.tar

        echo "gradcam"
        python3 get_passive_attention_unlabeled_cifar.py -o ../out/new_maps -m gradcam --data $IMAGEDIR --arch $ARCH --resume pretrained-cifar-$ARCH/model_best.pth.tar

        echo "scorecam"
        python3 get_passive_attention_unlabeled_cifar.py -o ../out/new_maps -m scorecam --data $IMAGEDIR --arch $ARCH --resume pretrained-cifar-$ARCH/model_best.pth.tar
    done
done
