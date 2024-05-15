import os

import cv2
import torch
from torch import nn
from torch.utils.data import Dataset


class v4Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.annotations = self._load_annotations()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_path, label_path = self.annotations[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self._load_label(label_path)
        return image, label

    def _load_annotations(self):
        image_paths = sorted(os.listdir(self.image_dir))
        label_paths = sorted(os.listdir(self.label_dir))
        return list(zip(image_paths, label_paths))

    def _load_label(self, label_path):
        with open(os.path.join(self.label_dir, label_path), 'r') as file:
            labels = file.readlines()
        # Process the labels as needed
        # For example, if the labels are bounding box coordinates, you might want to convert them to tensors
        # and normalize them by the image size
        return labels

