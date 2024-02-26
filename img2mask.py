import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
import torchvision
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchvision.transforms.functional as F
from pprint import pprint as pretty_print
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms.functional import convert_image_dtype


class Img2Mask:
    def __init__(self, img_dir, mask_dir) \
            -> None:
        """
        :param img_dir: path to directory containing images to be converted to masks
        :param mask_dir: path to directory containing masks to be converted from images
        :param img_size: size of images to be converted to masks (default: 256)
        """
        self._img_dir = img_dir
        self._mask_dir = mask_dir

        self.img_files_list = os.listdir(self._img_dir)
        self._img_size = len(self.img_files_list)
        self.mask_list = []

        self.init_model()

    @property
    def img_dir(self) -> str:
        """
        :return: path to directory containing images to be converted to masks
        """
        return self._img_dir

    @img_dir.setter
    def img_dir(self, img_dir) -> None:
        """
        :param img_dir: path to directory containing images to be converted to masks
        :return: None
        """
        self._img_dir = img_dir

    @property
    def mask_dir(self) -> str:
        """
        :return: path to directory containing masks to be converted from images
        """
        return self._mask_dir

    @mask_dir.setter
    def mask_dir(self, mask_dir) -> None:
        """
        :param mask_dir: path to directory containing masks to be converted from images
        """
        self._mask_dir = mask_dir

    def init_model(self) -> None:
        self.model = fcn_resnet50(pretrained=True, progress=False)
        self.model = self.model.eval()

    def img2mask(self):
        """
        Converts images to masks
        :return: None
        """
        pretty_print(f'Количество файлов в папке с изображениями: {self._img_size}')

        for img_file in self.img_files_list:
            current_img = cv2.imread(self._img_dir + img_file)
            torch_img = torch.from_numpy(cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            batch_int = torch.stack([torch_img])
            batch = convert_image_dtype(batch_int, dtype=torch.float)
            normalized_batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            output = self.model(normalized_batch)['out']
            normalized_masks = torch.nn.functional.softmax(output, dim=1)

        return None



img2mask = Img2Mask('./data/train/images', '.')
img2mask.img2mask()