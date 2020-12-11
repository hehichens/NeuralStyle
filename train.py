"""
trian model
edit by hichens
"""

import time
import torch
import torchvision
import os 
from utils.options import opt
from utils.utils import *
from utils.visualizer import Visualizer


## net 
net = create_model()


## Hyper Prameter
batch_size = opt.batch_size


## visulizer
visualizer = Visualizer()
total_iter = 0 
iter_time = time.time()
losses = []


## Result
checkpoint_dir = os.path.join(opt.checkpoint_dir, opt.model)
images_dir = os.path.join(opt.images_dir, opt.model)
mkdirs([checkpoint_dir, images_dir])


print("Training on: {}".format(opt.device))
for epoch in range(opt.num_epoch):
    if opt.model == 'BaseModel':
        data_loader = [(None, None)]
    elif opt.model == 'FST':
        data_loader = load_image_datasets(batch_size=batch_size)

    dataset_size = len(data_loader)
    net.set_input()
    for batch_id, (x, _) in enumerate(data_loader):
        total_iter += 1

        net.forward(x)
        net.optimize_parameters()

        # Print and plot training infomation
        if total_iter % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses.append(net.get_loss())
            loss_dict = {'total_loss': losses[-1]}

            visualizer.plot(total_iter, losses, names=['total_loss'])
            visualizer.print(epoch, loss_dict, time.time() - iter_time)
            iter_time = time.time()

        # Save the checkpoint
        if total_iter % opt.save_epoch_freq == 0:
            net.save_networks('latest', checkpoint_dir)
            net.save_networks(epoch+1, checkpoint_dir)

        # Display the result image
        if total_iter % opt.display_freq == 0:
            out_img_path =  os.path.join(images_dir, "{}_{}.png".format(opt.model, epoch+1))
            torchvision.utils.save_image(net.get_image(), out_img_path)
            visualizer.display_image(out_img_path)

