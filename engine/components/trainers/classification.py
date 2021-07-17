import torch

from .return_tuples import epoch_results
from ...utils import json_reader
from ...utils import errors
from ...utils import device
from ...parser.json import training_args_parser


class ClassificationTrainer:
    def __init__(self, model, train_loader, valid_loader, loss_fn, optimizer, hyperparameter_spec_file):
        self.__model = model
        self.__train_loader = train_loader
        self.__valid_loader = valid_loader
        self.__loss_fn = loss_fn
        self.__optimizer = optimizer
        self.__hyperparameter_spec_file = hyperparameter_spec_file

        self.__hyperparameter_spec = None
        self.__training_args = self.__parse_training_args()

    def __read_training_args_from_json(self):
        return json_reader.JSONReader.read_json(
            json_file_path=self.__hyperparameter_spec_file,
            error_class=errors.HyperparameterFileNotFound,
            error_message=f"{self.__hyperparameter_spec_file} not found, make sure you pass correct hyperparameter specification file!"
        )

    def __parse_training_args(self):
        self.__hyperparameter_spec = self.__read_training_args_from_json()

        training_dict = self.__hyperparameter_spec.pop('training')
        training_params = training_args_parser.TrainingArgsParser.parse_training_args(training_dict=training_dict)

        return training_params

    def __train_one_epoch(self, verbose=False):
        train_device = device.get_device()
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        self.__model.train()

        for i, data in enumerate(self.__train_loader):
            inputs, labels = data['img'], data['label']
            inputs = inputs.to(train_device)
            labels = labels.to(train_device)

            outputs = self.__model(inputs)
            loss = self.__loss_fn.loss_func(outputs, labels)

            batch_loss = loss.item()
            epoch_loss += batch_loss

            self.__optimizer.optimizer.zero_grad()
            loss.backward()
            self.__optimizer.optimizer.step()

            preds = torch.max(outputs, dim=1).indices
            batch_accuracy = (preds == labels).sum()
            epoch_accuracy += batch_accuracy

            if verbose:
                print(f"Batch: [{i+1}/{len(self.__train_loader)}], Loss: {loss.item()}, Accuracy: {batch_accuracy / len(inputs)}")

        epoch_loss /= len(self.__train_loader.dataset)
        epoch_accuracy /= len(self.__train_loader.dataset)

        return epoch_results(epoch_loss=epoch_loss, epoch_accuracy=epoch_accuracy)

    def __validate_one_epoch(self, verbose=False):
        valid_device = device.get_device()
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        self.__model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.__valid_loader):
                inputs, labels = data['img'], data['label']
                inputs = inputs.to(valid_device)
                labels = labels.to(valid_device)

                outputs = self.__model(inputs)
                loss = self.__loss_fn.loss_func(outputs, labels)

                batch_loss = loss.item()
                epoch_loss += batch_loss

                preds = torch.max(outputs, dim=1).indices
                batch_accuracy = (preds == labels).sum()
                epoch_accuracy += batch_accuracy

                if verbose:
                    print(f"Batch: [{i+1}/{len(self.__valid_loader)}], Loss: {loss.item()}, Accuracy: {batch_accuracy / len(inputs)}")

        epoch_loss /= len(self.__valid_loader.dataset)
        epoch_accuracy /= len(self.__valid_loader.dataset)

        return epoch_results(epoch_loss=epoch_loss, epoch_accuracy=epoch_accuracy)

    def __train_validate_one_epoch(self):
        training_results = self.__train_one_epoch()
        validation_results = self.__validate_one_epoch()

        return training_results, validation_results

    def train(self, verbose=False):
        for epoch_num in range(self.__training_args.epochs):
            training_results, validtion_results = self.__train_validate_one_epoch()

            if verbose:
                print(f"Epoch: [{epoch_num+1} / {self.__training_args.epochs}]")
                print(f"    Training Results: {training_results}")
                print(f"    Validation Results: {validtion_results}")