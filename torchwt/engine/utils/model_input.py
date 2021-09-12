from dataclasses import dataclass


@dataclass
class ModelInput:
    model_spec_file: str
    hyperparameter_spec_file: str