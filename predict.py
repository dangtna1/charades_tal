import torch
from utils.eval import predict_video
from data.transforms import get_default_transform
from models.r3d import build_r3d_model

filtered_actions = ["c047", "c046", "c048", "c049", "c050", "c051"]
cls2idx = {cls: idx for idx, cls in enumerate(filtered_actions)}
idx2cls = {v: k for k, v in cls2idx.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_r3d_model(len(filtered_actions)).to(device)
# model.load_state_dict(torch.load("the_model.pth"))  # when having a pre-built model
transform = get_default_transform()

preds = predict_video(
    model,
    "IPZIV",
    "CharadesEgo_v1_rgb/CharadesEgo_v1_rgb",
    transform,
    class_map=idx2cls,
    device=device,
)
print(preds)
