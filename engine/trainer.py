import torch
from tqdm import tqdm

from .utils import errors
from .utils import json_reader
from .utils import device
from .parser.json import training_args_parser


class Trainer:
    def __init__(self, model, train_loader, valid_loader, loss_fn, optimizer, hyperparameter_spec_file=None):
        self.__model = model
        self.__train_loader = train_loader
        self.__valid_loader = valid_loader
        self.__loss_fn = loss_fn
        self.__optimizer = optimizer
        self.__hyperparameter_spec_file = hyperparameter_spec_file

        self.__device = device.get_device()
        self.__hyperparameter_spec = None
        self.__hyperparameters = self.__parse_hyperparameters()

    def __read_hyperparameters_from_json(self):
        return json_reader.JSONReader.read_json(
            json_file_path=self.__hyperparameter_spec_file,
            error_class=errors.HyperparameterFileNotFound,
            error_message=f"{self.__hyperparameter_spec_file} not found, make sure you pass correct hyperparameter specification file!"
        )

    def __parse_hyperparameters(self):
        self.__hyperparameter_spec = self.__read_hyperparameters_from_json()

        training_dict = self.__hyperparameter_spec.pop("training")
        hyperameters = training_args_parser.TrainingArgsParser.parse_training_args(training_dict=training_dict)

        return hyperameters

    def train(self):
        self.__model.train()

        for epoch_num in tqdm(range(self.__hyperparameters['epochs'])):
            epoch_loss = 0.0
            for i, data in enumerate(self.__train_loader):
                input_data, target = data['input_data'], data['target']
                input_data = input_data.to(self.__device)
                target = target.to(self.__device)

                self.__optimizer.zero_grad()

                output = self.__model(input_data)

                loss = self.__loss_fn(output, target)
                epoch_loss += loss.item()

                loss.backward()
                self.__optimizer.step()

            print(f"Epoch: [{epoch_num + 1}/{self.__hyperparameters['epochs']}], Loss: {epoch_loss}")

    def validate(self):
        self.__model.eval()

        valid_loss = 0.0
        for i, data in enumerate(self.__valid_loader):
            input_data, target = data['input_data'], data['target']
            input_data = input_data.to(self.__device)
            target = target.to(self.__device)

            with torch.no_grad():
                output = self.__model(input_data)

            loss = self.__loss_fn(output, target)
            valid_loss += loss.item()

            loss.backward()
            self.__optimizer.step()

    def save_model(self, save_path):
        torch.save(self.__model.state_dict(), save_path)

    