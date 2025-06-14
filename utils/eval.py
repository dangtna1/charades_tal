import os
import glob
import torch
import numpy as np
from PIL import Image
import yaml

# Load config.yaml
with open("charades_tal/configs/config.yaml", "r") as f:
    configs = yaml.safe_load(f)


@torch.no_grad()
def predict_dual_view_video(
    model,
    ego_video_id,
    exo_video_id,
    frame_root=configs["frame_root"],
    clip_len=48,
    stride=24,
    fps=24,
    class_map=None,
    device="cpu",
    merge=True,
):
    model.eval()

    def load_clip_frames(video_id):
        dirp = os.path.join(frame_root, video_id)
        files = sorted(glob.glob(os.path.join(dirp, f"{video_id}-*.jpg")))
        return files

    ego_files = load_clip_frames(ego_video_id)
    exo_files = load_clip_frames(exo_video_id)

    clips = []
    for i in range(0, min(len(ego_files), len(exo_files)) - clip_len, stride):
        ego_tensor = (
            torch.stack(
                [
                    ds.transform(Image.open(ego_files[i + f]).convert("RGB"))
                    for f in range(clip_len)
                ],
                1,
            )
            .unsqueeze(0)
            .to(device)
        )

        exo_tensor = (
            torch.stack(
                [
                    ds.transform(Image.open(exo_files[i + f]).convert("RGB"))
                    for f in range(clip_len)
                ],
                1,
            )
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            prob = model(ego_tensor, exo_tensor).softmax(1).detach().cpu().numpy()[0]
            clips.append((i, prob))

    # for each clip, pick top class
    segs = [(i / fps, (i + clip_len) / fps, np.argmax(p)) for i, p in clips]

    if merge:
        # merge consecutive with same pred
        merged = []
        last = segs[0]
        for s, e, c in segs[1:]:
            if c == last[2]:
                last = (last[0], e, c)
            else:
                merged.append(last)
                last = (s, e, c)
        merged.append(last)
        segs = merged

    if class_map:
        return [(s, e, class_map[c]) for s, e, c in segs]

    return segs
