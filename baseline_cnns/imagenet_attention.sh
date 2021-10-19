#!/bin/bash

set -e

#for ARCH in "alexnet" "vgg16_bn" "resnet" "efficientnet"
for ARCH in "vit"
do
    for IMAGEDIR in "../all_images/new_selected" "../all_images/new_selected_2" "../all_images/new_selected_3" "../all_images/new_selected_4" "../all_images/next_25_selected" "../all_images/next_15_selected"
    do
        #echo "guidedbp"
        #python3 get_passive_attention_unlabeled_imagenet.py -o ../out/new_maps_imagenet -m guidedbp --data $IMAGEDIR --arch $ARCH

        #echo "guidedbpximage"
        #python3 get_passive_attention_unlabeled_imagenet.py -o ../out/new_maps_imagenet -m guidedbpximage --data $IMAGEDIR --arch $ARCH

        #echo "smoothgradguidedbp"
        #python3 get_passive_attention_unlabeled_imagenet.py -o ../out/new_maps_imagenet -m smoothgradguidedbp --data $IMAGEDIR --arch $ARCH

        #echo "gradcam"
        #python3 get_passive_attention_unlabeled_imagenet.py -o ../out/new_maps_imagenet -m gradcam --data $IMAGEDIR --arch $ARCH

        #echo "scorecam"
        #python3 get_passive_attention_unlabeled_imagenet.py -o ../out/new_maps_imagenet -m scorecam --data $IMAGEDIR --arch $ARCH

        echo "cameras"
        python3 get_passive_attention_unlabeled_imagenet.py -o ../out/new_maps_imagenet -m cameras --data $IMAGEDIR --arch $ARCH
    done
done
