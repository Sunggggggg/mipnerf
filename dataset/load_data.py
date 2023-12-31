import numpy as np
from .load_llff import load_llff_data
from .load_blender import load_blender_data


def load_data(datadir, dataset_type='blender', scale=4, 
              spherify=False, llffhold=8, no_ndc=False, # llff
              white_bkgd=True, testskip=8               # blender
              ):
    """
    images float 64
    """
    # Load data
    K = None
    if dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(datadir, scale,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if llffhold > 0:
            print('Auto LLFF holdout,', llffhold)
            i_test = np.arange(images.shape[0])[::llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

        return 

    elif dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(datadir, scale=scale, testskip=testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    else:
        print('Unknown dataset type', dataset_type, 'exiting')
    
    return images, poses, render_poses, hwf, K, near, far, i_train, i_val, i_test

