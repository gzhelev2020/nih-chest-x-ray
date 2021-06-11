import glob
import os
from functools import reduce
from itertools import chain
from typing import Any, Callable, List, Optional, Tuple

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
        # self._data_train is not passed as a parameter to prevent large memory usage
        self.filters = self._kfold_split(folds)
    
        print(len(self.filters))
        #print(self.filters[1])
        print((self._data_test))
        print(len(self._data_test))
        print(len(self.filters[4]))
        #print((self._data_train["patient"]))
        print(folds)
        
        #self.filters = self._self._kfold2_split(folds)
        
    #def _kfold_split2(self, folds: int, seed: int = 0) -> List[List[bool]]:
        '''
        use scikit learn sklearn.model_selection.GroupShuffleSplit
        Provides randomized train/test indices to split data according to a third-party provided group.
        in this case "patient". See doku on GroupShuffleSplit to finish the function.
        '''
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=folds, train_size=.7, random_state=0)
        groups = np.array(self._data_train["patient"])
        
        #!!
        #for train_idx, test_idx in gss.split(self._data_train, self._data_test, groups):
        #    print("TRAIN:", train_idx, "TEST:", test_idx)
        #!!

        #for some reason self._data_test is empty...
        #it is needed to run gss.split
        #so the the function can return the indexes for a specific split
        #when we have the indexes we can then turn them into booleans if this our goal

    def _kfold_split(self, folds: int, seed: int = 0) -> List[List[bool]]:
        '''
        Performs a k-fold split of self._data_train.
        A boolean filter list must be created for each split.
        A list of all these filters must be returned.
        I.e.,
            [[True, True, False, False],
             [False, False, True, True]]
        For a 2-fold split of a dataset of size 4

        The return value must be a list of :param:folds lists.
        Where each list must be precisely len(self._data_train) items long.
        There should be roughly the same number of elements in each split.
        Each index must be True exatly once (I.g., each data point is contained
          in exactly one split).
        It should be taken into account that all images of an individual
          patient are in the same split.
        The splits should be randomized but it must return identical results
          when the same seed is passed.
        '''

        # Currently just splits into :param:folds strides
        # I.e., [[True, True, False, False, False, False],
        #        [False, False, True, True, False, False],
        #        [False, False, False, False, True, True]]
        # Where the last first fold-1 items contains the same amout of dataset,
        # and the last item contains slighly more or less, depeneding on how
        # well len(self._data_train) is divisable by :param:folds
        items_per_fold = int(len(self._data_train)/folds)
        items_used = [False]*len(self._data_train)

        _filters = [None for x in range(folds)]
        for i in range(folds-1):
            _start = i * items_per_fold
            _end = _start + items_per_fold - 1

            fold_length = _end - _start + 1
            _filters[i] = [False]*len(self._data_train)

            for j in range(_start, _end+1):
                items_used[j] = True
                _filters[i][j] = True

        _filters[folds-1] = [not x for x in items_used]


        # Do not delete the following
        # makes sure all lists are of the same length
        _it = iter(_filters)
        _len = len(next(_it))
        if not all(len(l) == _len for l in _it):
            raise ValueError('not all lists have same length!')

        # makes sure each element is in at least one fold
        if not all(reduce(np.logical_or, _filters)):
            raise ValueError('some items are not in any fold')

        # makes sure each element is in exactly one fold
        if not all(reduce(np.logical_xor, _filters)):
            raise ValueError('some items are in more than one fold')

        return _filters


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
