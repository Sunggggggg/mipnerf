#!/bin/bash

object="lego hotdog materials chair drums ficus mic ship"

for obj in $object
do
    exp=blender_${obj}
    dir=/mnt2/SKY/dataset/nerf_synthetic/${obj}

    python run_multi_gpu_mipnerf.py --expname $exp --datadir $dir
done