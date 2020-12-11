"""
from the paper: https://arxiv.org/pdf/1603.08155.pdf
eidt by hichens
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from .Base import Base
import sys; sys.path.append("..")
from .NetWorks import TransformerNet, StyleVgg16
from utils.utils import *
from utils.options import opt


class FST(Base):
    def __init__(self):
        super(Base, self).__init__()
        self.style_image = load_image(opt.style_image_path).to(opt.device)
        self.content_image = load_image(opt.content_image_path).to(opt.device)
        self.transformer = TransformerNet().to(opt.device)
        self.vgg = StyleVgg16(requires_grad=False).to(opt.device)
        self.optimizer = optim.Adam(self.transformer.parameters(), opt.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.features_style = self.vgg(normalize_batch(self.style_image.repeat(opt.batch_size, 1, 1, 1)))
        self.Gram_style = [Gram(i) for i in self.features_style]
        self.model = {
            'transformer':self.transformer, 
            'vgg':self.vgg
        }

    def set_input(self):
        pass

    def forward(self, x):
        n_batch = len(x)
        x = x.to(opt.device)
        y = self.transformer(x)

        y = normalize_batch(y)
        x = normalize_batch(x)

        features_y = self.vgg(y)
        features_x = self.vgg(x)

        content_loss = self.loss_fn(features_x.relu2_2, features_y.relu2_2)

        style_loss = 0.0
        for ft_y, gm_s in zip(features_y, self.Gram_style):
            gm_y = Gram(ft_y)
            style_loss += self.loss_fn(gm_y, gm_s[:n_batch, :, :])
        self.total_loss = opt.alpha * content_loss + opt.beta * style_loss

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.total_loss.backward()
        self.optimizer.step()

    def get_loss(self):
        return self.total_loss.item()

    def get_image(self):
        return self.transformer(self.content_image)

if __name__ == "__main__":
    pass