import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from .Base import Base
import sys; sys.path.append("..")
from utils.utils import load_image, Gram
from utils.options import opt


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = torchvision.models.vgg19(pretrained=True).features
        self.choosen_features = [0, 5, 10, 19, 28]

    def forward(self, X):
        features = []
        for layer_id, layer in enumerate(self.model):
            X = layer(X)
            if layer_id in self.choosen_features:
                features.append(X)
        return features


class BaseModel(Base):
    def __init__(self):
        super(Base, self).__init__()
        self.net = VGG().to(opt.device).eval()
        self.content_image = load_image(opt.content_image_path).to(opt.device)
        self.style_image = load_image(opt.style_image_path).to(opt.device)
        self.out_image = self.content_image.clone().requires_grad_(True)
        self.optimizer = optim.Adam([self.out_image], opt.learning_rate)

        ## Hyper Pramaters
        self.alpha = opt.alpha
        self.beta = opt.beta
        self.model = {
            'basemodel':self.net
        }
        self.visual_names = ['total_loss']
    
    def set_input(self):
        pass

    def forward(self, x):
        content_features = self.net(self.content_image)
        style_features = self.net(self.style_image)
        out_features = self.net(self.out_image)

        content_loss, style_loss = 0, 0
        for content_feature, style_feature, out_feature in zip(
            content_features, style_features, out_features):
            # content loss 
            content_loss +=  torch.mean((content_feature - style_feature)**2)
            # style loss
            C = Gram(content_feature)
            S = Gram(style_feature)
            style_loss += torch.mean((S - C)**2)
        self.total_loss = self.alpha * content_loss + self.beta * style_loss

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.total_loss.backward()
        self.optimizer.step()

    def get_loss(self):
        return self.total_loss.item()

    def get_image(self):
        return self.out_image



if __name__ == "__main__":
    net = BaseModel()
    X = torch.rand(1, 3, 256, 256)
    print(net(X)[0].shape)