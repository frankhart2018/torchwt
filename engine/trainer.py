import torch


class Trainer:
    def __init__(self, model, train_set, valid_set):
        self.__model = model
        self.__train_set = train_set
        self.__valid_set = valid_set

    def train(self):
        pass

    def validate(self):
        pass

    