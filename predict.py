import torch
from utils.eval import predict_dual_view_video
from data.transforms import get_default_transform
from models.ego_exo_fusion import EgoExoFusionModel
import yaml

with open("charades_tal/configs/config.yaml", "r") as f:
    configs = yaml.safe_load(f)

filtered_actions = configs["filtered_actions"]
cls2idx = {cls: idx for idx, cls in enumerate(filtered_actions)}
idx2cls = {v: k for k, v in cls2idx.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EgoExoFusionModel(len(filtered_actions)).to(device)
model.load_state_dict(
    torch.load("charades_tal/models/ego_exo_fusion_model.pth")
)  # when having a pre-built model
transform = get_default_transform()

preds = predict_dual_view_video(
    model,
    "TVEHJEGO",
    "TVEHJ",
    configs["frame_root"],
    transform,
    class_map=idx2cls,
    device=device,
)
print(preds)
