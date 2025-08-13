import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from typing import Dict


class BaselineRes50Model(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        # load pre-trained Conv2D model
        self.model = resnet50(pretrained=True)

        # change input channels number to match the rasterizer's output
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        self.model.conv1 = nn.Conv2d(
            num_in_channels,
            self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False,
        )
        # change output size to (X, Y) * number of future states
        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    def forward(self, data):
        return self.model(data)
    
    def forward_pass(self, data, device, criterion):
        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)
        # Forward pass
        outputs = self.model(inputs).reshape(targets.shape)
        loss = criterion(outputs, targets)
        # loss = criterion(outputs, targets, target_availabilities)
        # not all the output steps are valid, but we can filter them out from the loss using availabilities
        loss = loss * target_availabilities
        loss = loss.mean()
        return loss, outputs

