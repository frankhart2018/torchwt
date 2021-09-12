import os
import glob

from torch.utils.data import DataLoader, random_split

from ..loader import Loader
from .dataset import ImageClassificationDataset
from ....utils import errors


class ImageClassificationLoader(Loader):
    def __init__(self, hyperparameter_spec_file, image_dir, im_extensions=[]):
        super().__init__(hyperparameter_spec_file=hyperparameter_spec_file)

        self.__image_dir = image_dir
        self.__allowed_extensions = list(set(["jpg", "jpeg", "png", "gif"] + im_extensions))
        self.__n_classes = None
        self.__classes_to_idx = None
        self.__idx_to_classes = None

    def __get_files_of_extension(self, path, extension):
        return glob.glob(os.path.join(path, f"*.{extension}"))

    def __get_dataloader_from_dataset(self, dataset, batch_size, shuffle=False):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_dataloader(self):
        file_paths = []

        dirs = os.listdir(self.__image_dir)
        dirs = [dir for dir in dirs if os.path.isdir(os.path.join(self.__image_dir, dir))]

        for dir in dirs:
            for allowed_extension in self.__allowed_extensions:
                file_paths += self.__get_files_of_extension(path=os.path.join(self.__image_dir, dir), extension=allowed_extension)

        if len(file_paths) == 0:
            raise errors.NoFileForExtensionFound(message=f"No files found with these extensions: [{self.__allowed_extensions}] in directory: {self.__image_dir}")

        dataset = ImageClassificationDataset(file_paths=file_paths)
        self.__n_classes = dataset.n_classes
        self.__classes_to_idx = dataset.classes_to_idx
        self.__idx_to_classes = dataset.idx_to_classes

        num_train_examples = round(len(dataset) * self.loader_spec.split_ratio)
        num_valid_examples = len(dataset) - num_train_examples

        train_dataset, valid_dataset = random_split(dataset, [num_train_examples, num_valid_examples])
        
        train_dataloader = self.__get_dataloader_from_dataset(
            dataset=train_dataset,
            batch_size=self.loader_spec.batch_size,
            shuffle=self.loader_spec.shuffle,
        )
        valid_dataloader = self.__get_dataloader_from_dataset(
            dataset=valid_dataset,
            batch_size=self.loader_spec.batch_size,
            shuffle=False,
        )

        return train_dataloader, valid_dataloader, len(file_paths)

    @property
    def n_classes(self):
        return self.__n_classes

    @property
    def classes_to_idx(self):
        return self.__classes_to_idx

    @property
    def idx_to_classes(self):
        return self.__idx_to_classes