import torch

from engine.components.model import Model
from engine.components.loss import Loss
from engine.components.optimizer import Optimizer
from engine.components.dataloaders.image_classification.dataloader import ImageClassificationLoader
from engine.components.trainers.classification import ClassificationTrainer


if __name__ == "__main__":
    m = Model(model_spec_file="engine/resources/model.json")

    # print(m)

    l = Loss(hyperparameter_spec_file="engine/resources/hyperparameters.json")
    # # print(l.loss_func)

    o = Optimizer(model=m, hyperparameter_spec_file="engine/resources/hyperparameters.json")
    # # print(o.optimizer)

    icl = ImageClassificationLoader(hyperparameter_spec_file="engine/resources/hyperparameters.json", image_dir="test-dir")
    train_loader, valid_loader, num_files = icl.get_dataloader()

    trainer = ClassificationTrainer(
        model=m, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        loss_fn=l, 
        optimizer=o,
        hyperparameter_spec_file="engine/resources/hyperparameters.json",
    )
    trainer.train(verbose=True)