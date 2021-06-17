import torch

from model import Model


if __name__ == "__main__":
    m = Model(model_spec_file="model.json")

    print(m)

    inp = torch.randn((1, 1))
    print(m(inp))