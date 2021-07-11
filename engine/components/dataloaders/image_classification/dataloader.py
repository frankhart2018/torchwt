import os
import glob

from ..loader import Loader
from .dataset import ImageClassificationDataset
from ....utils import errors


class ImageClassificationLoader(Loader):
    def __init__(self, hyperparameter_spec_file, image_dir):
        super().__init__(hyperparameter_spec_file=hyperparameter_spec_file)

        self.__image_dir = image_dir
        self.__allowed_extensions = ["jpg", "jpeg", "png", "gif"]
        self.__n_classes = None
        self.__classes_to_idx = None
        self.__idx_to_classes = None

    def __get_files_of_extension(self, extension):
        return glob.glob(os.path.join(self.__image_dir, f"*.{extension}"))

    def get_dataloader(self, im_extensions=[]):
        if im_extensions != []:
            self.__allowed_extensions += im_extensions
            self.__allowed_extensions = list(set(self.__allowed_extensions))

        file_paths = []

        for allowed_extension in self.__allowed_extensions:
            file_paths += self.__get_files_of_extension(extension=allowed_extension)

        if len(file_paths) == 0:
            raise errors.NoFileForExtensionFound(message=f"No files found with these extensions: [{self.__allowed_extensions}] in directory: {self.__image_dir}")

        dataset = ImageClassificationDataset(file_paths=file_paths)
