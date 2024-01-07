#!/bin/bash

object="lego hotdog materials chair drums ficus mic ship"

for obj in $object
do
    exp=blender_${obj}
    nerf_config=configs/NeRF/${obj}_8shot.txt
    mae_config=configs/MAE/mae.txt
    
    python run_mask_nerf.py --expname $exp --nerf_config $nerf_config --mae_config $mae_config --i_print 1
done