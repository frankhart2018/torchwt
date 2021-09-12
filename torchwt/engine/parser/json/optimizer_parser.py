import torch.optim as optim


OPTIMIZER_MAPPING = {
    "adam": optim.Adam,
}

class OptimizerParser:
    @staticmethod
    def parse_optimizer(model, optimizer_dict):
        optimizer_type = optimizer_dict.pop('type')
        optimizer_args = optimizer_dict.pop('args')

        optimizer_lr = optimizer_args.pop('lr')

        optimizer = OPTIMIZER_MAPPING[optimizer_type](model.parameters(), lr=optimizer_lr)

        return optimizer