from pathlib import Path
import torch
from torch import nn
import timm
from efficientnet_pytorch import EfficientNet
from .constants import NUM_CLASSES, DEVICE

def load_resnest50(checkpoint_path, num_classes=NUM_CLASSES):
    net = timm.create_model("tf_efficientnet_b4", pretrained=False)
    net.classifier = nn.Linear(net.classifier.in_features, num_classes)
    dummy_device = torch.device("cpu")
    d = torch.load(checkpoint_path, map_location=dummy_device)
    for key in list(d.keys()):
        d[key.replace("model.", "")] = d.pop(key)
    net.load_state_dict(d)
    net = net.to(DEVICE)
    net = net.eval()
    return net

def load_effnetb3(checkpoint_path, num_classes=NUM_CLASSES):
    #cf. https://www.kaggle.com/andradaolteanu/ii-shopee-model-training-with-pytorch-x-rapids
    net = EfficientNet.from_name("efficientnet-b3").cuda()
    if hasattr(net, "fc"):
        nb_ft = net.fc.in_features
        net.fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(net, "_fc"):
        nb_ft = net._fc.in_features
        net._fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(net, "classifier"):
        nb_ft = net.classifier.in_features
        net.classifier = nn.Linear(nb_ft, num_classes)
    elif hasattr(net, "last_linear"):
        nb_ft = net.last_linear.in_features
        net.last_linear = nn.Linear(nb_ft, num_classes)
    dummy_device = torch.device("cpu")
    d = torch.load(checkpoint_path, map_location=dummy_device)
    for key in list(d.keys()):
        d[key.replace("model.", "")] = d.pop(key)
    net.load_state_dict(d)
    net = net.to(DEVICE)
    net = net.eval()
    return net

def load_wsl(
    name:str,
    checkpoint_path:Path,
    num_classes=NUM_CLASSES
):
    net = torch.hub.load("facebookresearch/WSL-Images", name)
    if hasattr(net, "fc"):
        nb_ft = net.fc.in_features
        net.fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(net, "_fc"):
        nb_ft = net._fc.in_features
        net._fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(net, "classifier"):
        nb_ft = net.classifier.in_features
        net.classifier = nn.Linear(nb_ft, num_classes)
    elif hasattr(net, "last_linear"):
        nb_ft = net.last_linear.in_features
        net.last_linear = nn.Linear(nb_ft, num_classes)
    dummy_device = torch.device("cpu")
    d = torch.load(checkpoint_path, map_location=dummy_device)
    for key in list(d.keys()):
        d[key.replace("model.", "")] = d.pop(key)
    net.load_state_dict(d)
    net = net.to(DEVICE)
    net = net.eval()
    return net

def get_model(name, num_classes=NUM_CLASSES):
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
    if "resnest" in name:
        model = getattr(resnest_torch, name)(pretrained=True)
    elif "wsl" in name:
        model = torch.hub.load("facebookresearch/WSL-Images", name)
    elif name.startswith("resnext") or  name.startswith("resnet"):
        model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=True)
    elif name.startswith("tf_efficientnet_b"):
        model = getattr(timm.models.efficientnet, name)(pretrained=False)
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
