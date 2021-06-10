import torch
import torch.nn as nn
from torchvision import models


class Net(nn.Module):
    def __init__(self, num_ftrs):
        super(Net, self).__init__()
        self.fc = nn.Linear(num_ftrs, 15)

    def forward(self, x):
        x = self.fc(x)
        return x

model_ft = models.resnet50(pretrained=False)
# utils.set_parameter_requires_grad(model_ft, False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = Net(num_ftrs)
input_size = 224

model_ft.load_state_dict(torch.load("resnet_50_ft.pt"))
model.eval()



