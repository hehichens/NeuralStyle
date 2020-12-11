"""
print and visualize trianing infomation
edit by hichens
"""


import numpy as np
import os
import sys
import ntpath
import time
import visdom
from PIL import Image
from .utils import *
from . import html
from subprocess import Popen, PIPE
from .options import opt

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    def __init__(self):
        self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)

    def display_image(self, out_img_path):
        img = Image.open(out_img_path)
        loader = T.Compose([
            T.Resize((opt.image_size, opt.image_size)),
            T.ToTensor()
        ])
        img = loader(img)
        self.vis.image(
            img, 
            win='image',
            opts=dict(title='output image', caption='output image'),
        )


    def plot(self, total_iter, losses, names):
        """
        Pramaters:
            losses: loss dict
        """
        self.vis.line(X=[_ for _ in range(total_iter)], Y=losses, win='train', \
            opts={
                'title': opt.model + '--train',
                'legend': names,
                'showlegend': True, 
                })

    def print(self, epoch, loss_dict, time=0):
        print("Epoch: {}".format(epoch+1), end=', ')
        for k, v in loss_dict.items():
            print("%s: %.4f"%(k, v), end=', ')
        print("time: %.2f sec"%(time))

