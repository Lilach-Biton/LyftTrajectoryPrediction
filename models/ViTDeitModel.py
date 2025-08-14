import timm
import torch.nn as nn

class ViTDeitModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_input_channels = 3 + num_history_channels
        num_outputs = 2 * cfg["model_params"]["future_num_frames"]

        self.backbone = timm.create_model(
            "deit_small_patch16_224",
            pretrained=True,
            num_classes=0,
            in_chans=num_input_channels
        )
        
        embed_dim = self.backbone.num_features
        self.fc = nn.Linear(embed_dim, num_outputs)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)
    
    def forward_pass(self, data, device, criterion):
        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)

        # Forward through backbone
        feats = self.backbone(inputs)  # shape (B, embed_dim)
        
        # Project to desired output size
        outputs = self.fc(feats)  # shape (B, T*2)
        
        # Reshape to (B, T, 2)
        outputs = outputs.view(targets.size(0), targets.size(1), 2)

        loss = criterion(outputs, targets)
        # not all the output steps are valid, but we can filter them out from the loss using availabilities
        loss = loss * target_availabilities
        loss = loss.mean()
        return loss, outputs
    
