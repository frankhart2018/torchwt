class TrainingArgsParser:
    def parse_training_args(training_dict):
        hyperparameters = training_dict.pop('args')

        return hyperparameters