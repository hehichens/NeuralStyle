"""
some utils
edit by hichens
"""

from PIL import Image
import torchvision.transforms as T
from torchvision import datasets as D
from torch.utils.data import DataLoader
import sys
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
def load_image_datasets():
    loader = T.Compose([
        T.Resize(opt.image_size),
        T.CenterCrop(opt.image_size),
        T.ToTensor(),
        T.Lambda(lambda x: x.mul(255))
    ])
    datasets = D.ImageFolder(opt.image_dir, loader)
    data_loader = DataLoader(datasets, batch_size=opt.batch_size)
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

