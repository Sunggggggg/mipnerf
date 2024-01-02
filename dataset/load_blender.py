import os
import random
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
from einops import rearrange
from PIL import Image

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_blender_data(basedir, scale=4, testskip=1):
    """ Load nerf_synthetic dataset (specific object e.g. lego)
        For NeRF training inputs

    Return
        imgs                : [N, H, W, 3]
        poses               : [N, 4, 4]
        render_poses        : [40, 4, 4]
        [H, W, focal]       : []
        i_split             : [i_train, i_val, i_test]
    """
    splits = ['train', 'val', 'test']  #['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))   # Extrinsic matrix
        imgs = (np.array(imgs) / 255.).astype(np.float32)       # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    if scale:
        H = H//scale
        W = W//scale
        focal = focal/scale

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    return imgs, poses, render_poses, [H, W, focal], i_split
    
def load_nerf_synthetic_data(basedir, num_inputs=25, scale=4, testskip=8, white_bkgd=True):
    """ Load nerf_synthetic dataset with all objects
        For MAE training input
    Return

    imgs        # [O, N, H, W, 3]
    poses       # [O, N, 4, 4]
    """
    nerf_synthetic_dir = os.path.join(basedir, 'nerf_synthetic')
    object_list = os.listdir(nerf_synthetic_dir)
    object_list = [obj for obj in object_list if os.path.isdir(os.path.join(nerf_synthetic_dir, obj))]
    
    # Same input image size
    train_imgs, train_poses, val_imgs, val_poses, test_imgs, test_poses = [], [], [], [], [], []
    for _object in object_list :
        objectdir = os.path.join(nerf_synthetic_dir, _object)
        images, poses, render_poses, hwf, i_split = load_blender_data(objectdir, scale=scale, testskip=testskip)
        i_train, i_val, i_test = i_split
    
        if white_bkgd :
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else :
            images = images[...,:3]
        
        # train
        train_imgs.append(images[i_train])        # [N, H, W, 3]
        train_poses.append(poses[i_train])      # [N, 4, 4]

        # val
        val_imgs.append(images[i_val])        # [N, H, W, 3]
        val_poses.append(poses[i_val])      # [N, 4, 4]

        # test
        test_imgs.append(images[i_test])        # [N, H, W, 3]
        test_poses.append(poses[i_test])      # [N, 4, 4]

    train_imgs = np.stack(train_imgs, 0)      # [O, N, H, W, 3]    
    train_poses = np.stack(train_poses, 0)    # [O, N, 4, 4]

    val_imgs = np.stack(val_imgs, 0)      # [O, N, H, W, 3]    
    val_poses = np.stack(val_poses, 0)    # [O, N, 4, 4]

    test_imgs = np.stack(test_imgs, 0)      # [O, N, H, W, 3]    
    test_poses = np.stack(test_poses, 0)    # [O, N, 4, 4]

    return train_imgs, train_poses, val_imgs, val_poses, test_imgs, test_poses, hwf, object_list

def sampling_pose(N, theta_range, phi_range, radius_range) :
    """ sampling with sorting angle
    """
    theta = np.random.uniform(*theta_range, N)
    phi = np.random.uniform(*phi_range, N)
    radius = np.random.uniform(*radius_range, N)

    render_poses = torch.stack([pose_spherical(theta[i], phi[i], radius[i]) for i in range(N)], 0)    # [N, 4, 4]
    render_poses = render_poses.cpu().numpy()
    return render_poses