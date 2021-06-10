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


class ChestXRayImageDataset(VisionDataset):
    rel_label_file = 'Data_Entry_2017.csv'
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

        if not train:
            frac = frac * 0.2
        self.data, self.targets = self._load_data(frac)


    def _load_data(self, frac: float = 1.) -> Tuple[Any, Any]:
        df = pd.read_csv(self.label_file,
                         usecols=['Image Index', 'Finding Labels',])

        if(frac < 1):
            df = df.sample(frac=frac)

        df.rename(columns = {
            'Image Index': 'idx',
            'Finding Labels': 'findings'
        }, inplace = True)

        # replace 'No Finding' with none
        df['findings'] = df['findings'].map(lambda x: x.replace('No Finding',
                                                                'none'))

        # | split labels to list
        df['findings'] = df['findings'].map(lambda x: x.split('|')).tolist()

        for label in self.labels:
            if len(label) > 0:
                df[label] = df['findings'].map(lambda finding: 1.0 if label in finding else 0.0)

        return df.iloc[:, 0], df.iloc[:, 2:17]

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
