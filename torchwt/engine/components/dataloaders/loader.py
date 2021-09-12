from json import load
from ...utils import errors
from ...utils import json_reader
from ...parser.json import loader_parser


class Loader:
    def __init__(self, hyperparameter_spec_file):
        self.__hyperparameter_spec_file = hyperparameter_spec_file
        
        self.__hyperparameters_spec = None
        self.__loader_spec = self.__parse_loader_from_json()

    def __read_loader_from_json(self):
        return json_reader.JSONReader.read_json(
            json_file_path=self.__hyperparameter_spec_file,
            error_class=errors.HyperparameterFileNotFound,
            error_message=f"{self.__hyperparameter_spec_file} not found, make sure you pass correct hyperparameter specification file!"
        )

    def __parse_loader_from_json(self):
        self.__hyperparameters_spec = self.__read_loader_from_json()

        loader_dict = self.__hyperparameters_spec.pop('dataset')
        loader_spec = loader_parser.LoaderParser.parse_loader(loader_dict=loader_dict)

        return loader_spec

    @property
    def loader_spec(self):
        return self.__loader_spec