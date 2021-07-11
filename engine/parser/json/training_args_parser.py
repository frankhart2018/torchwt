from collections import namedtuple


class TrainingArgsParser:
    def parse_training_args(training_dict):
        hyperparameters = training_dict.pop('args')

        epochs = hyperparameters.pop('epochs')
        training_spec_tuple = namedtuple("TrainingSpecTuple", ["epochs"])
        training_spec = training_spec_tuple(epochs=epochs)

        return training_spec