"""
Hyper Parameters options
edit by hichens
"""

import argparse
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def options():
    parser = argparse.ArgumentParser()
    
    ## Global varialbe
    parser.add_argument("--device", help="select device: cpu or gpu, not support for muti gpu", \
        default=device, type=str)
    parser.add_argument("--alpha", help="basemodel alpha value", \
        default=1, type=float)
    parser.add_argument("--beta", help="basemodel beta value", \
        default=1e2, type=float)
    parser.add_argument("--seed", help="random seed", \
        default=42, type=int)
    

    ## Data
    # content image
    parser.add_argument("--content_image_path", help="content image path", \
        default="datasets/content-images/anyhathway.jpg", type=str)
    parser.add_argument("--style_image_path", help="style image path", \
        default="./datasets/style-images/candy.jpg", type=str)
    # style image
    parser.add_argument("--content_image_dir", help="content image dir", \
        default="./datasets/content-images", type=str)
    parser.add_argument("--style_image_dir", help="style image dir", \
        default="./datasets/style-images", type=str)
    # traning image
    parser.add_argument("--image_dir", help="training image directory", \
        default="./datasets/images", type=str)
    # image parameter
    parser.add_argument("--image_size", help="input image size", \
        default=256, type=int)
    # load data
    parser.add_argument("--batch_size", help="load data batch size", \
        default=4, type=int)
    


    ## Train
    parser.add_argument("--model", help="model name", \
        default="BaseModel", type=str)
    parser.add_argument("--num_epoch", help="number of epochs", \
        default=2, type=int)
    parser.add_argument("--learning_rate", help="learning rate", \
        default=1e-3, type=float)
    
    ## Result
    # Net
    parser.add_argument("--checkpoint_dir", help="checkpoints saved directory", \
        default="./weights", type=str)
    parser.add_argument("--save_epoch_freq", help="save epoch frequency", \
        default=100, type=int)
    # Image
    parser.add_argument("--images_dir", help="image saved directory", \
        default="./results/images", type=str)


    ## Visualize
    parser.add_argument("--print_freq", help="print trianing information frequency", \
        default=10, type=int)
    parser.add_argument("--display_port", help="visdom display port", \
        default=8096, type=int)
    parser.add_argument("--display_winsize", help="visdom diaplay window size", \
        default=400, type=int)
    parser.add_argument("--use_html", help="wheather use html", \
        default=True, type=bool)
    

    return parser.parse_args()


opt = options()