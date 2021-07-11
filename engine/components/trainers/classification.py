import torch

from .return_tuples import epoch_results


class ClassificationTrainer:
    def __init__(self, model, train_loader, valid_loader, loss_fn, optimizer):
        self.__model = model
        self.__train_loader = train_loader
        self.__valid_loader = valid_loader
        self.__loss_fn = loss_fn
        self.__optimizer = optimizer

    def __get_device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_one_epoch(self, verbose=False):
        device = self.__get_device()
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for i, data in enumerate(self.__train_loader):
            inputs, labels = data['img'], data['label']
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = self.__model(inputs)
            loss = self.__loss_fn(labels, outputs)

            batch_loss = loss.item()
            epoch_loss += batch_loss

            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            preds = torch.max(outputs, dim=1)
            batch_accuracy = preds == labels
            epoch_accuracy += batch_accuracy

            if verbose:
                print(f"Batch: [{i+1}/{len(self.__train_loader)}], Loss: {loss.item()}, Accuracy: {batch_accuracy}")

        epoch_loss /= len(self.__train_loader.dataset)
        epoch_accuracy /= len(self.__train_loader.dataset)

        return epoch_results(epoch_loss=epoch_loss, epoch_accuracy=epoch_accuracy)

    def __validate_one_epoch(self, verbose=False):
        pass