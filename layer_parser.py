import torch.nn as nn

from collections import OrderedDict

from torch.nn.modules import activation


LAYER_MAPPING = {
    "linear": nn.Linear,
    "conv2d": nn.Conv2d,
}

ACTIVATION_MAPPING = {
    "relu": nn.ReLU,
}

class LayerParser:
    @staticmethod
    def parse_layer(layer_dict, layer_num):
        layer_type = layer_dict.pop('type')
        layer_args = layer_dict.pop('args')
        layer_activation = layer_args.pop('activation')
        
        layers = OrderedDict()

        layers[f"{layer_type}-{layer_num}"] = LAYER_MAPPING[layer_type](**layer_args)

        if layer_activation != "none":
            layers[f"{layer_type}-{layer_num}-{layer_activation}"] = ACTIVATION_MAPPING[layer_activation]()

        return layers