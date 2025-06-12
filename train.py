import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import optim, nn
import yaml

from data.dataset import CharadesTAL
from data.transforms import get_default_transform
from models.r3d import build_r3d_model

# Load config.yaml
with open("charades_tal/configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load annotation & preprocess
anno_df = pd.read_csv("CharadesEgo/CharadesEgo_v1_train.csv")  # Change dir if necessary

# Preprocess the annotation
rows = []
for _, r in anno_df.iterrows():
    vid = r["id"]
    actions = r["actions"]
    if isinstance(actions, str):
        for trip in actions.split(";"):
            cls, st, en = trip.split()
            rows.append(
                {"video": vid, "class": cls, "start": float(st), "end": float(en)}
            )
all_actions = pd.DataFrame(rows)

# Sample some actions
filtered_actions = ["c047", "c046", "c048", "c049", "c050", "c051"]

# Sample some videos for the actions
filtered = all_actions[all_actions["class"].isin(filtered_actions)][:20]

cls2idx = {cls: idx for idx, cls in enumerate(filtered_actions)}

# Instantiate dataset
dataset = CharadesTAL(
    filtered,
    config["frame_root"],
    transform=get_default_transform(),
    cls2idx=cls2idx,
    clip_len=config["clip_len"],
    stride=config["stride"],
    fps=config["fps"],
)
loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_r3d_model(len(filtered_actions)).to(device)
opt = optim.Adam(model.parameters(), lr=config["lr"])
crit = nn.CrossEntropyLoss()

# Training loop
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        logits = model(clips)
        loss = crit(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {total_loss/len(loader):.4f}")
