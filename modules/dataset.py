import glob
import os
from itertools import chain
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.io import read_image

class ChestXRayImages():
    rel_label_file = 'Data_Entry_2017.csv'
    rel_test_list = 'test_list.txt'
    rel_img_dir = 'images_*/images'

    data_train = None
    data_test = None

    filters = []

    def __init__(
        self,
        root: str,
        folds: int
    ):
        _data = None
        _test_files = None
        test_filter = None

        # load the entire data csv file
        _data = pd.read_csv(
            os.path.join(root, self.rel_label_file),
            usecols=['Image Index', 'Finding Labels', 'Patient ID']
        )

        _data.rename(columns = {
            'Image Index': 'idx',
            'Finding Labels': 'findings',
            'Patient ID': 'patient'
        }, inplace = True)

        _data = self._preprocess_data(_data)

        _test_files = pd.read_csv(
            os.path.join(root, self.rel_test_list),
            header=None,
            squeeze=True
        )

        # split test/train data
        test_filter = pd.Index(_data['idx']).isin(_test_files)
        self.data_test = _data.loc[test_filter].reset_index(drop=True)
        self.data_train = _data.loc[[not x for x in test_filter]].reset_index(drop=True)

        # perform k-fold train data split
        # just splits into k strides
        items_per_fold = int(len(self.data_train)/folds)
        items_used = [False]*len(self.data_train)

        self.filters = [None for x in range(folds)]
        for i in range(folds-1):
            _start = i * items_per_fold
            _end = _start + items_per_fold - 1

            fold_length = _end - _start + 1
            self.filters[i] = [False]*len(self.data_train)

            for j in range(_start, _end+1):
                items_used[j] = True
                self.filters[i][j] = True

        self.filters[folds-1] = [not x for x in items_used]


    def get_fold(self, fold: int):
        return self.data_train.loc[self.filters[fold]].reset_index(drop=True)


    def _preprocess_data(self, _data):
        # replace 'No Finding' with none
        _data['findings'] = _data['findings'].map(lambda x: x.replace('No Finding',
                                                                      'none'))

        # | split labels to list
        _data['findings'] = _data['findings'].map(lambda x: x.split('|')).tolist()

        return _data

class ChestXRayImageDataset(VisionDataset):
    rel_label_file = 'Data_Entry_2017.csv'
    rel_test_list = 'test_list.txt'
    rel_img_dir = 'images_*/images'

    labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
              'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
              'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
              'Pneumothorax', 'none']

    def __init__(
        self,
        root: str,
        train: bool = True,
        frac: float = 1.,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super(ChestXRayImageDataset,
              self).__init__(root, transform=transform,
                             target_transform=target_transform)

        self.img_dir = os.path.join(root, self.rel_img_dir)
        self.label_file = os.path.join(self.root, self.rel_label_file)
        self.test_list = os.path.join(self.root, self.rel_test_list)


        self.data, self.targets = self._load_data(frac)

    def _filter_data(data, subset):
        pass


    def _load_data(self, frac: float = 1.) -> Tuple[Any, Any]:
        data = pd.read_csv(self.label_file,
                         usecols=['Image Index', 'Finding Labels',])


        if(frac < 1):
            data = data.sample(frac=frac)

        data.rename(columns = {
            'Image Index': 'idx',
            'Finding Labels': 'findings'
        }, inplace = True)

        # replace 'No Finding' with none
        data['findings'] = data['findings'].map(lambda x: x.replace('No Finding',
                                                                    'none'))

        # | split labels to list
        data['findings'] = data['findings'].map(lambda x: x.split('|')).tolist()

        for label in self.labels:
            if len(label) > 0:
                data[label] = data['findings'].map(lambda finding: 1.0 if label in finding else 0.0)

        return data.iloc[:, 0], data.iloc[:, 2:17]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = os.path.join(self.img_dir, self.data.iloc[index])
        img_path = glob.glob(img_path)
        img = Image.open(img_path[0]).convert('RGB')

        target = torch.from_numpy(self.targets.iloc[index].to_numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
