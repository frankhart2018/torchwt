import torch.nn as nn


LOSS_MAPPING = {
    "mse": nn.MSELoss,
}

class LossParser:
    @staticmethod
    def parse_loss(loss_dict):
        loss_function_type = loss_dict.pop('type')
        loss_function = LOSS_MAPPING[loss_function_type]()

        return loss_function