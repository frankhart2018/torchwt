from torch.utils.data import Dataset
import cv2
import os
import torch


class ObjectDetectionDataset(Dataset):
    def __init__(self, file_paths):
        self.__file_paths = file_paths

        self.__classes = None
        self.__n_classes = None
        self.__classes_to_idx = {}
        self.__idx_to_classes = {}

        self.__initialize_class_metadata()

    def __initialize_class_metadata(self):
        self.__classes = [os.path.split(os.path.dirname(file_path))[1] for file_path in self.__file_paths]
        self.__n_classes = len(set(self.__classes))
        
        for idx, class_ in enumerate(set(self.__classes)):
            self.__classes_to_idx[class_] = idx
            self.__idx_to_classes[idx] = class_

    @property
    def n_classes(self):
        return self.__n_classes

    @property
    def classes_to_idx(self):
        return self.__classes_to_idx

    @property
    def idx_to_classes(self):
        return self.__idx_to_classes

    def __len__(self):
        return len(self.__file_paths)

    def __getitem__(self, idx):
        file_path = self.__file_paths[idx]
        label = self.__classes_to_idx[self.__classes[idx]]

        img = cv2.imread(filename=file_path)
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        img = cv2.flip(src=img, flipCode=1)
        
        img = torch.FloatTensor(img).permute(2, 1, 0)

        return {
            "img": img,
            "label": torch.tensor(label),
        }