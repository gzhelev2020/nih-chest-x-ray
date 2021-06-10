import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchvision import transforms

from modules import net, trainer
from modules.dataset import ChestXRayImageDataset

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type = str, help = 'Path to training data')
    parser.add_argument('--model-path', type = str, help = 'Path to store models')
    parser.add_argument('--test-bs', type = int, default = 64, help = 'test batch size')
    parser.add_argument('--val-bs', type = int, default = 64, help = 'val batch size')
    parser.add_argument('--train-bs', type = int, default = 64, help = 'train batch size')
    parser.add_argument('--lr', type = float, default = 1e-5, help = 'Learning Rate passed to optimizer')
    parser.add_argument('--device', type = str, default = device, help = 'Force usage of device')
    parser.add_argument('--epochs', type = str, default = 2, help = 'Train for n epochs')
    parser.add_argument('--log-interval', type = int, default = 5, help = 'log every n batches')
    parser.add_argument('--save-interval', type = int, default = 5, help = 'save every n batches')
    parser.add_argument('--data-frac', type = float, default = 1, help = 'use only fraction of the data')
    args = parser.parse_args()

    data_train = ChestXRayImageDataset(args.data_path, True, transform=transform,
                                       frac=args.data_frac)
    data_test = ChestXRayImageDataset(args.data_path, False, transform=transform,
                                      frac=args.data_frac)


    model = net.get_model(len(ChestXRayImageDataset.labels))

    summary(model, input_size=(args.train_bs, 3, 244, 244))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr = args.lr)

    print('Using device: {}'.format(args.device))

    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.test_bs)
    val_loader = torch.utils.data.DataLoader(data_test,
                                             batch_size=args.val_bs)
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.train_bs)

    loss_fn = nn.BCEWithLogitsLoss()

    trainer.run(device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                model=model,
                epochs=args.epochs,
                loss_fn=loss_fn,
                optimizer=optimizer,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                labels=data_train.labels,
                lr=args.lr,
                model_dir=args.model_path)

if __name__ == "__main__":
    main()
