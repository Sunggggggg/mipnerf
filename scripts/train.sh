#!/bin/bash

#object="lego hotdog materials chair drums ficus mic ship"
object="hotdog materials chair drums ficus mic ship"

for obj in $object
do
    exp=blender_${obj}
    dir=/mnt2/SKY/dataset/nerf_synthetic/${obj}
    mae_config=configs/MAE/mae.txt

    python run_mipnerf.py --expname $exp --datadir $dir --mae_config $mae_config 
done