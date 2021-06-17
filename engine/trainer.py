import torch


class Trainer:
    def __init__(self, model, train_set, valid_set, loss_fn, optimizer, hyperparameter_spec_file=None):
        self.__model = model
        self.__train_set = train_set
        self.__valid_set = valid_set
        self.__loss_fn = loss_fn
        self.__optimizer = optimizer
        self.__hyperparameter_spec_file = hyperparameter_spec_file

    def train(self):
        pass

    def validate(self):
        pass

    