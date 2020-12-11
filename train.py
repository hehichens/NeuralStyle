"""
trian model
edit by hichens
"""

import time
import torch
import torchvision
import os 
from utils.options import opt
from utils.utils import create_model
from utils.visualizer import Visualizer


## net 
net = create_model()


## visulizer
visualizer = Visualizer()
total_iters = 0                # the total number of training iterations
optimize_time = 0.1
times = []


print("Training on: {}".format(opt.device))
for epoch in range(opt.num_epoch):
    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0
    visualizer.reset()

    net.set_input()
    net.forward()
    net.optimize_parameters()

    ## Visualize
    if (epoch + 1) % opt.print_freq == 0:
        train_loss = net.get_loss()
        print("Epoch: %d, Training loss: %.4f, Time: %d sec"%(epoch+1, train_loss, time.time() - epoch_start_time))
        img_save_path = os.path.join(opt.images_dir, "{}_{}.png".format(opt.model, epoch+1))
        print("=> save image at: {}".format(img_save_path))
        torchvision.utils.save_image(net.get_image(), img_save_path)
        epoch_start_time = time.time()

    ## Save checkpoint
    if (epoch + 1) % opt.save_epoch_freq == 0:
        net.save_networks('latest')
        net.save_networks(epoch+1)