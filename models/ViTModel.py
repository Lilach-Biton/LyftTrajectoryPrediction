import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from typing import Dict

class ViTModel(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        # load pre-trained ViT model
        self.model = vit_b_16(weights="IMAGENET1K_V1")

        # change input channels number to match the rasterizer's output
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        old_proj = self.model.conv_proj

        self.model.conv_proj = nn.Conv2d(
            num_in_channels,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=False,
        )

        # change output size to (X, Y) * number of future states
        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_targets)

    def forward(self, data):
        return self.model(data)
    
    def forward_pass(self, data, device, criterion):
        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)
        # Forward pass
        outputs = self.model(inputs).reshape(targets.shape)
        loss = criterion(outputs, targets)
        # not all the output steps are valid, but we can filter them out from the loss using availabilities
        loss = loss * target_availabilities
        loss = loss.mean()
        return loss, outputs
    
