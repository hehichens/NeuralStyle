"""
This class is an abstract base class (ABC) for models.
edit by hichens
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
import sys; sys.path.append("..")
from utils.options import opt
import os
import torch

class Base(ABC):
    def __init__(self):
        self.checkpoint_dir = os.path.join(opt.checkpoint_dir, opt.model)
        self.model = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []

    @staticmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def save_networks(self, epoch):
        for name, net in self.model.items():
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(opt.checkpoint_dir, save_filename)
            print("=>save checkpoint at: {}".format(save_path))
            if opt.device == 'cuda':
                torch.save(net.module.cpu().state_dict(), save_path)
                net.to(opt.device)
            else:
                torch.save(net.cpu().state_dict(), save_path)
        
    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)  # return object attribute value 
        return visual_ret