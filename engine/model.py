from typing import OrderedDict
import torch
import torch.nn as nn

import json
import os

from .utils import errors
from .utils import json_reader
from .layer_parser import LayerParser


class Model(nn.Module):
    def __init__(self, model_spec_file, hyperparams_spec_file=None):
        super(Model, self).__init__()

        self.__model_spec_file = model_spec_file
        self.__hyperparams_spec_file = hyperparams_spec_file

        self.__model_spec = None
        self.__hyperparams_spec = None
        self.__model = self.__parse_model_from_json()

    def __read_model_from_json(self):
        if not os.path.exists(self.__model_spec_file):
            raise errors.ModelFileNotFound(message=f"{self.__model_spec_file} not found, make sure you pass correct model specification file!")

        return json_reader.JSONReader.read_json(json_file_path=self.__model_spec_file)

    def __parse_model_from_json(self):
        self.__model_spec = self.__read_model_from_json()

        model_ordered_dict = OrderedDict()

        for layer_num, layer_dict in enumerate(self.__model_spec):
            model_ordered_dict.update(LayerParser.parse_layer(layer_dict=layer_dict, layer_num=layer_num))

        model = nn.Sequential(
            model_ordered_dict
        )

        return model

    def forward(self, input):
        return self.__model(input)