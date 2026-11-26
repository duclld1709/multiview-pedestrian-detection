"""Stride-eight dilated ResNet-18 used by BEVNet."""
import torch.nn as nn
import torchvision


def _dilate(layer, first, rest):
    """Turn a stride-2 ResNet stage into stride 1 with dilated 3x3 convs (torchvision's scheme)."""
    for i, block in enumerate(layer):
        if i == 0:
            block.conv1.stride = (1, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        for conv, d in ((block.conv1, first if i == 0 else rest), (block.conv2, rest)):
            conv.dilation, conv.padding = (d, d), (d, d)


def resnet18(pretrained):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    _dilate(model.layer3, 1, 2)
    _dilate(model.layer4, 2, 4)
    return nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                         model.layer1, model.layer2, model.layer3, model.layer4)
