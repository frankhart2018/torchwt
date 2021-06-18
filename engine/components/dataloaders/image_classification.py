

from .loader import Loader


class ImageClassificationLoader(Loader):
    def __init__(self, hyperparameter_spec_file, image_dir):
        super().__init__(hyperparameter_spec_file=hyperparameter_spec_file)

        self.__image_dir = image_dir

    def get_dataloader(self):
        pass