from ..utils import errors
from ..utils import json_reader
from ..parser.json import loss_parser

class Loss:
    def __init__(self, hyperparameter_spec_file):
        self.__hyperparameter_spec_file = hyperparameter_spec_file

        self.__hyperparameter_spec = None
        self.__loss_func = self.__parse_loss()

    def __read_loss_from_json(self):
        return json_reader.JSONReader.read_json(
            json_file_path=self.__hyperparameter_spec_file,
            error_class=errors.HyperparameterFileNotFound,
            error_message=f"{self.__hyperparameter_spec_file} not found, make sure you pass correct hyperparameter specification file!"
        )

    def __parse_loss(self):
        self.__hyperparameter_spec = self.__read_loss_from_json()

        loss_dict = self.__hyperparameter_spec.pop('loss')
        loss_function = loss_parser.LossParser.parse_loss(loss_dict=loss_dict)

        return loss_function

    @property
    def loss_func(self):
        return self.__loss_func