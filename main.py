import argparse, os
import config
import utils
import net, trainer

import torch
from Dataset import ChestXRayImageDataset
import torch.optim as optim
import torch.nn as nn


DATA_DIR = './data/'
device = 'cpu'


data_train = ChestXRayImageDataset(DATA_DIR, True, transform=config.transform,
                                   frac=0.001)
data_test = ChestXRayImageDataset(DATA_DIR, False, transform=config.transform,
                                  frac=0.001)
data_train[0]

model = net.get_model(len(ChestXRayImageDataset.labels))
print('There are {} Million trainable parameters in the {} model'.format(
    utils.count_parameters(model), model.__class__.__name__
))

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr = config.LR)

test_loader = torch.utils.data.DataLoader(data_test, batch_size=1)
val_loader = torch.utils.data.DataLoader(data_test, batch_size=1)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=1)

loss_fn = nn.BCEWithLogitsLoss()
os.mkdir('./models')
trainer.run(device, train_loader, val_loader, test_loader, model, 2, loss_fn,
            optimizer, 1, 1, data_train.labels, 1e-3, 1, './models')

