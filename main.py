import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchvision import transforms

from modules import net, trainer
from modules.dataset import ChestXRayImageDataset, ChestXRayImages


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type = str, help = 'Path to training data')
    parser.add_argument('--model-path', type = str, help = 'Path to store models')
    parser.add_argument('--test-bs', type = int, default = 64, help = 'test batch size')
    parser.add_argument('--val-bs', type = int, default = 64, help = 'val batch size')
    parser.add_argument('--train-bs', type = int, default = 64, help = 'train batch size')
    parser.add_argument('--lr', type = float, default = 1e-5, help = 'Learning Rate passed to optimizer')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'Force usage of device')
    parser.add_argument('--epochs', type = str, default = 2, help = 'Train for n epochs')
    parser.add_argument('--log-interval', type = int, default = 5, help = 'log every n batches')
    parser.add_argument('--save-interval', type = int, default = 5, help = 'save every n batches')
    parser.add_argument('--data-frac', type = float, default = 1, help = 'use only fraction of the data')
    parser.add_argument('--folds', type=int, default=5, help='how many folds to produce')
    parser.add_argument('--fold-id', type=int, default=0, help='Which fold id to use for test/val split')
    parser.add_argument('--seed', type=int, default=0, help='Seed the random generator to get reproducability')
    args = parser.parse_args()

    if args.device == 'tpu':
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    elif args.device == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')


    data_wrapper = ChestXRayImages(args.data_path, folds=args.folds, frac=args.data_frac)    #list csv dateien
    data_train = ChestXRayImageDataset(
        args.data_path,
        data_wrapper.data_train(args.fold_id),                   #data wrapper
        transform=transform
    )
    data_val = ChestXRayImageDataset(                             #data val
        args.data_path,
        data_wrapper.data_val(args.fold_id),
        transform=transform
    )
    data_test = ChestXRayImageDataset(                           #data test
        args.data_path,
        data_wrapper.data_test,
        transform=transform
    )

    return

    model = net.get_model(len(ChestXRayImageDataset.labels))

    summary(model, input_size=(args.train_bs, 3, 244, 244))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr = args.lr)

    print('Using device: {}'.format(device))
    print('With {} Test datasets, {} val data sets and {} train datasets'.format(
        len(data_test), len(data_val), len(data_train)
    ))

    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.test_bs)
    val_loader = torch.utils.data.DataLoader(data_val,
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
