import os
import glob
import torch
import numpy as np
from PIL import Image


@torch.no_grad()
def predict_video(
    model,
    video_id,
    frame_root,
    transform,
    clip_len=48,
    stride=24,
    fps=24,
    class_map=None,
    device="cpu",
    merge=True,
):
    model.eval()
    dirp = os.path.join(frame_root, video_id)
    # gather all frames in memory
    files = sorted(glob.glob(os.path.join(dirp, f"{video_id}-*.jpg")))
    # load & preprocess
    clips = []
    for i in range(0, len(files) - clip_len, stride):
        tensor = (
            torch.stack(
                [
                    transform(Image.open(files[i + 1 + f]).convert("RGB"))
                    for f in range(clip_len)
                ],
                1,  # (C, T, H, W)
            )
            .unsqueeze(0)
            .to(device)
        )
        prob = model(tensor).softmax(1).detach().cpu().numpy()[0]  # or remove detach()?
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
