"""
some utils
edit by hichens
"""

from PIL import Image
import torchvision.transforms as T
from torchvision import datasets as D
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
from imutils import paths
import importlib
from .options import opt

## load image
def load_image(filename=None, image_size=opt.image_size):
    image = Image.open(filename)
    loader = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])
    image = loader(image).unsqueeze(0) # (channels, width, height) ==> (batch_size=1, channels, width, height)
    return image


## load image datasets
def load_image_datasets(batch_size=opt.batch_size):
    loader = T.Compose([
        T.Resize(opt.image_size),
        T.CenterCrop(opt.image_size),
        T.ToTensor(),
        T.Lambda(lambda x: x.mul(255))
    ])
    datasets = D.ImageFolder(opt.image_dir, loader)
    data_loader = DataLoader(datasets, batch_size=batch_size)
    return data_loader



## Gram Matrix
def Gram(feature):
    batch_size, channel, height, width = feature.shape
    feature = feature.view(batch_size, channel, width*height)
    feature_t = feature.transpose(1, 2)
    gram = feature.bmm(feature_t) / (channel * height * width)
    return gram


## create model
def create_model():
    model = find_model_using_name(opt.model)
    instance = model()
    print("model [%s] was created" % type(instance).__name__)
    return instance


## find model according model name
def find_model_using_name(model_name):
    """Import the module "models/[model_name].py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


## normalize using imagenet mean and std
def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


## Converts a Tensor array into a numpy image array
def tensor2im(input_image, imtype=np.uint8):
    """

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


## Save a numpy image to the disk
def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


## Create empty directories if they don't exist
def mkdirs(paths):
    """
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

## Create a single empty directory if it didn't exist
def mkdir(path):
    """
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)