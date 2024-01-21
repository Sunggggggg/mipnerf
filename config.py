import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    # Basic
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='logs')
    parser.add_argument("--eval", action='store_true', help='Eval mode')
    parser.add_argument('--nerf_config', is_config_file=True, help='config file path')
    parser.add_argument("--nerf_weight", type=str, default=None)
    # Dataset
    parser.add_argument("--datadir", type=str, default='/mnt2/SKY/dataset/nerf_synthetic/lego')
    parser.add_argument("--dataset_type", type=str, default='blender')
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument("--testskip", type=int, default=8)
    # Optimizer and scheduler
    parser.add_argument("--lr_init", type=float, default=1e-3)      
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_final", type=float, default=5e-5)
    parser.add_argument("--lr_delay_steps", type=int, default=2500)
    parser.add_argument("--lr_delay_mult", type=float, default=0.1)
    # Loss
    parser.add_argument("--coarse_weight_decay", type=float, default=0.1)
    # Training hyperparams
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--netchunk", type=int, default=1024*64)
    parser.add_argument("--N_rand", type=int, default=1024*4, help='#of Sampling rays')
    parser.add_argument("--no_ndc", action='store_true')
    parser.add_argument("--precrop_iters", type=int, default=500)
    parser.add_argument("--precrop_frac", type=float, default=.5) 
    # Model 
    parser.add_argument("--use_viewdirs", action="store_true", help='use full 5D input instead of 3D')
    parser.add_argument("--randomized", action="store_true")
    parser.add_argument("--ray_shape", type=str, default="cone")        # should be "cylinder" if llff
    parser.add_argument("--white_bkgd", action="store_true")           # should be False if using llff
    parser.add_argument("--num_levels", type=int, default=2)
    parser.add_argument("--N_samples", type=int, default=64)
    parser.add_argument("--N_importance", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--density_noise", type=float, default=0.0)
    parser.add_argument("--density_bias", type=float, default=-1.0)
    parser.add_argument("--rgb_padding", type=float, default=0.001)
    parser.add_argument("--resample_padding", type=float, default=0.01)
    parser.add_argument("--min_deg", type=int, default=0)
    parser.add_argument("--max_deg", type=int, default=16)
    parser.add_argument("--viewdirs_min_deg", type=int, default=0)
    parser.add_argument("--viewdirs_max_deg", type=int, default=4)
    # Few-shot
    parser.add_argument("--nerf_input", type=int, default=8)
    parser.add_argument("--mae_input", type=int, default=25)
    # MAE
    parser.add_argument('--mae_config', is_config_file=True, help='config file path')
    parser.add_argument("--mae_weight", type=str, default=None)
    parser.add_argument("--emb_type", type=str, default="IMAGE")        # OR "PATCH"
    parser.add_argument("--image_token", type=int, default=16)
    parser.add_argument("--cam_pose_encoding", action='store_true')
    # MAE (Model)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=24)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--decoder_embed_dim', type=int, default=512)
    parser.add_argument('--decoder_depth',type=int, default=8, )
    parser.add_argument('--decoder_num_heads', type=int,default=16)
    # MAE (loss)
    parser.add_argument("--mae_loss_func", type=str, default="COSINE")  # OR "PERCE"
    parser.add_argument("--loss_lam_c", type=float, default=2.0)         # loss_lam_f * 0.1
    parser.add_argument("--loss_lam_f", type=float, default=2.0)        # 
    # logging/saving options
    parser.add_argument("--i_print", type=int, default=10)
    parser.add_argument("--i_weights", type=int, default=100)
    parser.add_argument("--i_testset", type=int, default=100)
    parser.add_argument("--i_video",   type=int, default=10000)
    
    return parser

def mae_args_parser():
    parser = configargparse.ArgumentParser()
    # Basic
    parser.add_argument("--basedir", default='./mae_logs', type=str)
    parser.add_argument('--expname', type=str, help= "Experiment name")
    parser.add_argument("--eval", action='store_true', help='Eval mode')
    parser.add_argument('--mae_config', is_config_file=True, help='config file path')
    parser.add_argument("--mae_weight", type=str, default=None)
    # Dataset
    parser.add_argument("--datadir", type=str, default='/mnt/SKY/dataset')
    parser.add_argument("--dataset_type", type=str, default='blender')
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument("--testskip", type=int, default=8)
    parser.add_argument('--mae_input', default=25, type=int, help= "#of MAE images")
    parser.add_argument("--white_bkgd", action='store_true')
    # Training hyperparams
    parser.add_argument('--epochs', default=5001, type=int)
    parser.add_argument('--lrate', default=0.0001, type=float)
    # Model
    parser.add_argument('--emb_type', default='IMAGE', type=str)
    parser.add_argument('--cam_pose_encoding', action='store_true')
    parser.add_argument('--image_token', default=16, type=int)
    parser.add_argument('--norm_pix_loss', action='store_true')
    # Encoder part
    parser.add_argument('--embed_dim', default=1024, type=int)
    parser.add_argument('--depth', default=24, type=int)
    parser.add_argument('--num_heads', default=16, type=int)
    # Decoder part
    parser.add_argument('--decoder_embed_dim', default=512, type=int)
    parser.add_argument('--decoder_depth', default=8, type=int)
    parser.add_argument('--decoder_num_heads', default=16, type=int)
    # logging/saving options
    parser.add_argument('--i_print', default=10, type=int)
    parser.add_argument('--i_figure', default=100, type=int)
    parser.add_argument('--i_weight', default=1000, type=int)
    
    return parser