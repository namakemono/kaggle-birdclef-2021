from pathlib import Path
import torch
from torch import nn
import timm
from resnest.torch import resnest50
from efficientnet_pytorch import EfficientNet
import pretrainedmodels

from .constants import NUM_CLASSES, DEVICE

def load_model(
    name:str,
    checkpoint_path:Path,
    num_classes=NUM_CLASSES
):
    net = get_model(name, num_classes, pretrained=False)
    dummy_device = torch.device("cpu")
    d = torch.load(checkpoint_path, map_location=dummy_device)
    for key in list(d.keys()):
        d[key.replace("model.", "")] = d.pop(key)
    net.load_state_dict(d)
    net = net.to(DEVICE)
    net = net.eval()
    return net

def get_model(name, num_classes=NUM_CLASSES, pretrained=True):
    """
    Loads a pretrained model.
    Supports ResNest, ResNext-wsl, EfficientNet, ResNext and ResNet.

    Arguments:
        name {str} -- Name of the model to load

    Keyword Arguments:
        num_classes {int} -- Number of classes to use (default: {1})

    Returns:
        torch model -- Pretrained model
    """
    if name == "resnest50":
        model = resnest50(pretrained=pretrained)
    elif name == "resnest50d_1s4x24d":
        model = timm.models.resnest50d_1s4x24d(pretrained=pretrained)
    elif "wsl" in name:
        model = torch.hub.load("facebookresearch/WSL-Images", name)
    elif name.startswith("resnext") or  name.startswith("resnet"):
        model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=pretrained)
    elif name.startswith("tf_efficientnet_b"):
        model = getattr(timm.models.efficientnet, name)(pretrained=pretrained)
    elif name.startswith("densenet"):
        model = getattr(timm.models.densenet, name)(pretrained=pretrained)
    elif "efficientnet-b" in name:
        model = EfficientNet.from_pretrained(name)
    else:
        model = pretrainedmodels.__dict__[name](pretrained='imagenet')

    if hasattr(model, "fc"):
        nb_ft = model.fc.in_features
        model.fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "_fc"):
        nb_ft = model._fc.in_features
        model._fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "classifier"):
        nb_ft = model.classifier.in_features
        model.classifier = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "last_linear"):
        nb_ft = model.last_linear.in_features
        model.last_linear = nn.Linear(nb_ft, num_classes)
    return model
