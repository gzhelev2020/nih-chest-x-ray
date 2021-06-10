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

    _data_train = None
    _data_test = None

    filters = []

    def __init__(
        self,
        root: str,
        folds: int,
        frac: float = 1
    ):
        _data = None
        _test_files = None
        test_filter = None

        # load the entire data csv file
        # Image Index: file name of the image
        # Finding Labels: | seperated findings
        # Patient ID: unique id for each patient. Needed for good splits
        _data = pd.read_csv(
            os.path.join(root, self.rel_label_file),
            usecols=['Image Index', 'Finding Labels', 'Patient ID']
        )

        _data = _data.sample(frac=frac)

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
        self._data_test = _data.loc[test_filter].reset_index(drop=True)
        self._data_train = _data.loc[[not x for x in test_filter]].reset_index(drop=True)

        # perform k-fold train data split
        # just splits into k strides
        items_per_fold = int(len(self._data_train)/folds)
        items_used = [False]*len(self._data_train)

        self.filters = [None for x in range(folds)]
        for i in range(folds-1):
            _start = i * items_per_fold
            _end = _start + items_per_fold - 1

            fold_length = _end - _start + 1
            self.filters[i] = [False]*len(self._data_train)

            for j in range(_start, _end+1):
                items_used[j] = True
                self.filters[i][j] = True

        self.filters[folds-1] = [not x for x in items_used]


    def _preprocess_data(self, _data):
        # replace 'No Finding' with none
        _data['findings'] = _data['findings'].map(lambda x: x.replace('No Finding',
                                                                      'none'))

        # | split labels to list
        _data['findings'] = _data['findings'].map(lambda x: x.split('|')).tolist()

        return _data

    @property
    def data_test(self):
        return self._data_test


    def data_val(self, fold_id: int):
        _data = self._data_train.loc[self.filters[fold_id]].reset_index(drop=True)
        return _data[['idx', 'findings']]

    def data_train(self, fold_id: int):
        _data = self._data_train.loc[[not x for x in self.filters[fold_id]]].reset_index(drop=True)
        return _data[['idx', 'findings']]


class ChestXRayImageDataset(VisionDataset):
    rel_img_dir = 'images_*/images'

    labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
              'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
              'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
              'Pneumothorax', 'none']

    def __init__(
        self,
        root: str,
        data_frame: pd.DataFrame,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super(ChestXRayImageDataset,
              self).__init__(root, transform=transform,
                             target_transform=target_transform)

        self.img_dir = os.path.join(root, self.rel_img_dir)

        for label in self.labels:
            data_frame[label] = data_frame['findings'].map(lambda finding: 1.0 if label in finding else 0.0)

        self.data = data_frame


    def _load_data(self, frac: float = 1.) -> Tuple[Any, Any]:
        return data.iloc[:, 0], data.iloc[:, 2:17]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = os.path.join(self.img_dir, self.data.iloc[index, 0])
        img_path = glob.glob(img_path)
        img = Image.open(img_path[0]).convert('RGB')

        target = torch.tensor(self.data.iloc[index, 2:17].values.astype(np.float32))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
