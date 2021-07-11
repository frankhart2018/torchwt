import torch

from engine.components.model import Model
from engine.components.loss import Loss
from engine.components.optimizer import Optimizer
from engine.components.dataloaders.image_classification.dataloader import ImageClassificationLoader
from engine.components.trainers.classification import ClassificationTrainer
from engine.utils import model_input


if __name__ == "__main__":
    model_input = model_input.ModelInput(
        model_spec_file="engine/resources/model.json",
        hyperparameter_spec_file="engine/resources/hyperparameters.json",
    )

    model = Model(model_spec_file=model_input.model_spec_file)

    loss = Loss(hyperparameter_spec_file=model_input.hyperparameter_spec_file)

    optimizer = Optimizer(model=model, hyperparameter_spec_file=model_input.hyperparameter_spec_file)

    icl = ImageClassificationLoader(hyperparameter_spec_file=model_input.hyperparameter_spec_file, image_dir="test-dir")
    train_loader, valid_loader, num_files = icl.get_dataloader()

    trainer = ClassificationTrainer(
        model=model, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        loss_fn=loss, 
        optimizer=optimizer,
        hyperparameter_spec_file=model_input.hyperparameter_spec_file,
    )
    trainer.train(verbose=True)