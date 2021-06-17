import torch

from engine.components.model import Model
from engine.components.loss import Loss
from engine.components.optimizer import Optimizer


if __name__ == "__main__":
    m = Model(model_spec_file="engine/resources/model.json")

    print(m)

    inp = torch.randn((1, 1))
    print(m(inp))

    l = Loss(hyperparameter_spec_file="engine/resources/hyperparameters.json")
    print(l.loss_func)

    o = Optimizer(model=m, hyperparameter_spec_file="engine/resources/hyperparameters.json")
    print(o.optimizer)