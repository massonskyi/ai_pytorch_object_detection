import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from config import MODEL_YAML


class SPPF(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(SPPF, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        spp_layers = [F.adaptive_max_pool2d(x, output_size=(size, size)) for size in self.pool_sizes]
        spp_layers.append(x)  # Original feature map without pooling
        x = torch.cat(spp_layers, dim=1)  # Concatenate along the channel dimension
        return x


class ConcatBlock(nn.Module):
    def __init__(self, in_channels):
        super(ConcatBlock, self).__init__()
        self.in_channels = in_channels

    def forward(self, x_list):
        x = torch.cat(x_list, dim=1)
        return x


class DetectBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DetectBlock, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.detect_layers = nn.ModuleList([
            nn.Conv2d(int(in_channels[-1] // 2), int(num_classes), kernel_size=1)
        ])

    def forward(self, x):
        for layer in self.detect_layers:
            x = layer(x)
        return x


class C2fBlock(nn.Module):
    def __init__(self, in_channels, use_batch_norm=True):
        super(C2fBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ]
        if use_batch_norm:
            layers.insert(1, nn.BatchNorm2d(in_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Backbone(nn.Module):
    def __init__(self, config, in_channels):
        super(Backbone, self).__init__()
        self.layers = nn.ModuleList()
        for layer_cfg in config['backbone']:
            self.layers.extend(_parse_layer(layer_cfg))

    def forward(self, x):
        if isinstance(x, list):
            for block in x:
                for layer in self.layers:
                    _x = layer(block)
                return _x
        else:
            x = torch.cat(x, dim=0)
            for layer in self.layers:
                x = layer(x)
            return x
def _parse_layer(layer_cfg):
    if layer_cfg[2] == 'Conv':
        return [nn.Conv2d(layer_cfg[3][1], layer_cfg[3][0], kernel_size=3, stride=layer_cfg[3][2], padding=1)
                for _ in range(layer_cfg[1])]
    elif layer_cfg[2] == 'nn.Upsample':
        return [nn.Upsample(scale_factor=layer_cfg[3][1], mode=layer_cfg[3][2])
                for _ in range(layer_cfg[1])]
    elif layer_cfg[2] == 'C2f':
        return [C2fBlock(layer_cfg[3][0], layer_cfg[3][0])
                for _ in range(layer_cfg[1])]
    elif layer_cfg[2] == 'Concat':
        return [ConcatBlock(layer_cfg[3][0])
                for _ in range(layer_cfg[1])]
    elif layer_cfg[2] == 'Detect':
        return [DetectBlock(layer_cfg[0], layer_cfg[3][0])
                for _ in range(layer_cfg[1])]  # Update to use num_classes from the config
    elif layer_cfg[2] == 'SPPF':
        return [SPPF(layer_cfg[3][1], layer_cfg[3][0]) for _ in range(layer_cfg[1])]


class Head(nn.Module):
    def __init__(self, config):
        super(Head, self).__init__()
        self.layers = nn.ModuleList()
        for layer_cfg in config['head']:
            self.layers.extend(_parse_layer(layer_cfg))

    def forward(self, x_list):
        for layer in self.layers:
            x_list = layer(x_list)
        return x_list


class DModel(nn.Module):
    def __init__(self, config_path=MODEL_YAML):
        super(DModel, self).__init__()
        try:
            with open(config_path) as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
        except Exception as e:
            raise e

        self.backbone = Backbone(config, in_channels=1)
        self.head = Head(config)

    def forward(self, images, targets=None):
        x = self.backbone(images)
        x = self.head(x)

        # If targets are provided, compute the losses
        if targets is not None:
            loss_dict = {}
            for i, target in enumerate(targets):
                # Extract the ground truth boxes and labels
                boxes = target['boxes']
                labels = target['labels']

                # Compute the classification loss
                cls_loss = F.cross_entropy(x[:, :, i, i], labels)
                loss_dict['cls_loss_{}'.format(i)] = cls_loss

                # Compute the box regression loss (e.g., smooth L1 loss)
                box_loss = F.smooth_l1_loss(x[:, :, i, :4], boxes)
                loss_dict['box_loss_{}'.format(i)] = box_loss

            # Sum the losses
            loss = sum(loss for loss in loss_dict.values())
            return loss_dict, loss

        # If targets are not provided, return the output features
        else:
            return x
