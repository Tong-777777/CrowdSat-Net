# this code is based on https://github.com/Alexandre-Delplanque/HerdNet/blob/main/animaloc/datasets/csv.py

import numpy as np
import torch
import os
import PIL
from PIL import Image
import numpy
import albumentations
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from datasets.register import DATASETS
from datasets.transforms import *


@DATASETS.register()
class CrowdDataset(Dataset):

    def __init__(
            self,
            data_root: str,
            train: bool = False,
            train_list: str = 'crowd_train.list',
            val_list: str = 'crowd_val.list',
            albu_transforms: Optional[list] = None,
            end_transforms: Optional[list] = None
    ) -> None:
        """
        :param data_root: path to the images folder
        :param train_list: list of training images, each line represent a raw image path and its corresponding label path
        :param val_list: list of validation images, each line represent a raw image path and its corresponding label path
        :param albu_transforms: an albumentations' transformations
                list that takes input sample as entry and returns a transformed
                version. Defaults to None.
        :param end_transforms: list of transformations that takes
                tensor and expected target as input and returns a transformed
                version. These will be applied after albu_transforms. Defaults
                to None.
        """
        assert isinstance(albu_transforms, (list, type(None))), \
            f'albumentations-transformations must be a list, got {type(albu_transforms)}'

        assert isinstance(end_transforms, (list, type(None))), \
            f'end-transformations must be a list, got {type(end_transforms)}'

        self.data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), data_root))
        self.train_list = train_list
        self.val_list = val_list
        self.albu_transforms = albu_transforms
        self.end_transforms = end_transforms

        if train:
            self.img_list_file = self.train_list.split(',')
        else:
            self.img_list_file = self.val_list.split(',')

        self._img_map = {}
        self._img_list = []

        self.to_tensor = transforms.ToTensor()

        for _, file_list in enumerate(self.img_list_file):
            file_list = file_list.strip()
            with open(os.path.join(self.data_root, file_list)) as fin:
                for line in fin:
                    if len(line) < 2:
                        continue
                    line = line.strip().split()
                    self._img_map[os.path.join(self.data_root, line[0].strip())] = \
                        os.path.join(self.data_root, line[1].strip())

        self._img_list = sorted(list(self._img_map.keys()))

    def _load_image(self, index: int) -> Image.Image:
        img_path = self._img_list[index]

        return PIL.Image.open(img_path).convert('RGB')

    def _load_target(self, index: int) -> Dict[str, List[Any]]:

        img_path = self._img_list[index]
        label_path = self._img_map[img_path]

        points = []
        labels = []
        with open(label_path) as files:
            for line in files:
                x = int(line.strip().split(' ')[0])
                y = int(line.strip().split(' ')[1])
                points.append([x, y])
                labels.append(1)

        target = {
            'img_id': index,
            'img_path': img_path,
            'points': points,
            'labels': labels
        }

        return target

    def _transforms(
            self,
            img: Image.Image,
            target: dict
            ) -> Tuple[torch.Tensor, dict]:

        label_fields = target.copy()
        for key in ['points', 'img_id', 'img_path']:
            label_fields.pop(key)

        if self.albu_transforms:
            transform_pipeline = albumentations.Compose(
                self.albu_transforms,
                keypoint_params=albumentations.KeypointParams(
                    format='xy',
                    label_fields=list(label_fields.keys())
                )
            )

            transformed = transform_pipeline(
                image=numpy.array(img),
                keypoints=target['points'],
                **label_fields
            )

            tr_image = numpy.asarray(transformed['image'])
            transformed.pop('image')

            transformed['points'] = transformed['keypoints']
            transformed.pop('keypoints')

            for key in ['img_id', 'img_path']:
                transformed[key] = target[key]

            tr_image, tr_target = SampleToTensor()(tr_image, transformed, 'point')

            if self.end_transforms is not None:
                for trans in self.end_transforms:
                    tr_image, tr_target = trans(tr_image, tr_target)

            return tr_image, tr_target
        else:
            return img, target


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        img = self._load_image(index)
        target = self._load_target(index)

        tr_img, tr_target = self._transforms(img, target)

        return tr_img, tr_target

    def __len__(self):
        return len(self._img_list)



if __name__ == '__main__':
    pass


