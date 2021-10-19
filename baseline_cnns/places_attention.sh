#!/bin/bash

set -e

for ARCH in "alexnet" "resnet50"
do
    for IMAGEDIR in "../all_images/new_selected" "../all_images/new_selected_2" "../all_images/new_selected_3" "../all_images/new_selected_4" "../all_images/next_25_selected" "../all_images/next_15_selected"
    do
        echo "guidedbp"
        python3 get_passive_attention_unlabeled_places.py -o ../out/new_maps_places -m guidedbp --data $IMAGEDIR --arch $ARCH --resume pretrained-places365-$ARCH/${ARCH}_places365.pth.tar

        echo "guidedbpximage"
        python3 get_passive_attention_unlabeled_places.py -o ../out/new_maps_places -m guidedbpximage --data $IMAGEDIR --arch $ARCH --resume pretrained-places365-$ARCH/${ARCH}_places365.pth.tar

        echo "smoothgradguidedbp"
        python3 get_passive_attention_unlabeled_places.py -o ../out/new_maps_places -m smoothgradguidedbp --data $IMAGEDIR --arch $ARCH --resume pretrained-places365-$ARCH/${ARCH}_places365.pth.tar

        echo "gradcam"
        python3 get_passive_attention_unlabeled_places.py -o ../out/new_maps_places -m gradcam --data $IMAGEDIR --arch $ARCH --resume pretrained-places365-$ARCH/${ARCH}_places365.pth.tar

        echo "scorecam"
        python3 get_passive_attention_unlabeled_places.py -o ../out/new_maps_places -m scorecam --data $IMAGEDIR --arch $ARCH --resume pretrained-places365-$ARCH/${ARCH}_places365.pth.tar

        echo "cameras"
        python3 get_passive_attention_unlabeled_places.py -o ../out/new_maps_places -m cameras --data $IMAGEDIR --arch $ARCH --resume pretrained-places365-$ARCH/${ARCH}_places365.pth.tar
    done
done
