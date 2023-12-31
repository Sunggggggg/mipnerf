import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    # Basic
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='logs')
    parser.add_argument("--eval", action='store_true', help='Eval mode')
    # Optimizer and scheduler
    parser.add_argument("--lr_init", type=float, default=1e-3)      
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_final", type=float, default=5e-5)
    parser.add_argument("--lr_delay_steps", type=int, default=2500)
    parser.add_argument("--lr_delay_mult", type=float, default=0.1)
    # Loss
    parser.add_argument("--coarse_weight_decay", type=float, default=0.1)
    # Training hyperparams
    parser.add_argument("--max_iters", type=int, default=200_000)
    parser.add_argument("--N_rand", type=int, default=1024*4, help='#of Sampling rays')
    
    # Model 
    parser.add_argument("--use_viewdirs", action="store_false")
    parser.add_argument("--randomized", action="store_false")
    parser.add_argument("--ray_shape", type=str, default="cone")        # should be "cylinder" if llff
    parser.add_argument("--white_bkgd", action="store_false")           # should be False if using llff
    parser.add_argument("--override_defaults", action="store_true")
    parser.add_argument("--num_levels", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--density_noise", type=float, default=0.0)
    parser.add_argument("--density_bias", type=float, default=-1.0)
    parser.add_argument("--rgb_padding", type=float, default=0.001)
    parser.add_argument("--resample_padding", type=float, default=0.01)
    parser.add_argument("--min_deg", type=int, default=0)
    parser.add_argument("--max_deg", type=int, default=16)
    parser.add_argument("--viewdirs_min_deg", type=int, default=0)
    parser.add_argument("--viewdirs_max_deg", type=int, default=4)


    # MISC
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float, default=.5, 
                        help='fraction of img taken for central crops') 
    return parser