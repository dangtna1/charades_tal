import torch.nn as nn
import torchvision.models.video as video_models


def build_r3d_model(num_classes):
    model = video_models.r3d_18(weights="KINETICS400_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
