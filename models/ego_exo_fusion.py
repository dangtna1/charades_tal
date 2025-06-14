import torch
import torch.nn as nn
import torchvision.models.video as video_models


class EgoExoFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ego_encoder = video_models.r3d_18(weights="KINETICS400_V1")
        self.exo_encoder = video_models.r3d_18(weights="KINETICS400_V1")

        # Freeze lower layers if needed
        # for param in self.ego_encoder.parameters(): param.requires_grad = False

        dim = self.ego_encoder.fc.in_features
        self.fc = nn.Linear(dim * 2, num_classes)
        self.ego_encoder.fc = nn.Identity()
        self.exo_encoder.fc = nn.Identity()

    def forward(self, ego, exo):
        ego_feat = self.ego_encoder(ego)  # (B, F)
        exo_feat = self.exo_encoder(exo)  # (B, F)
        fused = torch.cat([ego_feat, exo_feat], dim=1)
        return self.fc(fused)
