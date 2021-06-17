from ..utils import errors
from ..utils import json_reader
from ..parser.json import optimizer_parser

class Optimizer:
    def __init__(self, model, hyperparameter_spec_file):
        self.__model = model
        self.__hyperparameter_spec_file = hyperparameter_spec_file

        self.__hyperparameter_spec = None
        self.__optimizer = self.__parse_optimizer()

    def __read_optimizer_from_json(self):
        return json_reader.JSONReader.read_json(
            json_file_path=self.__hyperparameter_spec_file,
            error_class=errors.HyperparameterFileNotFound,
            error_message=f"{self.__hyperparameter_spec_file} not found, make sure you pass correct hyperparameter specification file!"
        )

    def __parse_optimizer(self):
        self.__hyperparameter_spec = self.__read_optimizer_from_json()

        optimizer_dict = self.__hyperparameter_spec.pop('optimizer')
        optimizer = optimizer_parser.OptimizerParser.parse_optimizer(model=self.__model, optimizer_dict=optimizer_dict)

        return optimizer

    @property
    def optimizer(self):
        return self.__optimizer