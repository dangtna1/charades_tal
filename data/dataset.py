import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class CharadesTAL(Dataset):
    def __init__(
        self,
        annots,
        frame_root,
        clip_len=48,
        stride=12,
        fps=24,
        transform=None,
        cls2idx=None,
    ):

        self.clip_len, self.stride, self.fps = clip_len, stride, fps
        self.transform = transform
        self.cls2idx = cls2idx
        self.samples = []

        for _, row in annots.iterrows():
            vid, cls, st, en = row["video"], row["class"], row["start"], row["end"]
            frame_dir = os.path.join(frame_root, vid)
            if not os.path.isdir(frame_dir):
                continue
            # compute frame indices
            start_f = int(st * fps)
            end_f = int(en * fps)
            # build sliding-window samples within this segment
            for f0 in range(start_f, end_f - clip_len, stride):
                self.samples.append(
                    {
                        "frame_dir": frame_dir,
                        "start_idx": f0,
                        "label": self.cls2idx[cls],
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        frames = []

        for idx in range(s["start_idx"], s["start_idx"] + self.clip_len):
            offset = 1
            idx = idx + offset  # frame indices are 1-based
            # frame filenames are zero-padded 6 digits
            path = os.path.join(
                s["frame_dir"], f"{os.path.basename(s['frame_dir'])}-{idx:06d}.jpg"
            )
            img = Image.open(path).convert("RGB")
            frames.append(self.transform(img))
        clip = torch.stack(frames, 1)  # (C, T, H, W)
        return clip, s["label"]
