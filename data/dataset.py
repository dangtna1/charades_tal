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

            # Get exo and ego videos
            if vid.endswith("EGO"):
                ego_vid = vid
                exo_vid = vid[:-3]
            else:
                ego_vid = vid + "EGO"
                exo_vid = vid

            ego_frame_dir = os.path.join(frame_root, ego_vid)
            exo_frame_dir = os.path.join(frame_root, exo_vid)

            if not os.path.isdir(ego_frame_dir or exo_frame_dir):
                continue  # continue if lack either one of perspective

            # compute frame indices
            start_f = int(st * fps)
            end_f = int(en * fps)

            # build sliding-window samples within this segment
            for f0 in range(start_f, end_f - clip_len, stride):
                self.samples.append(
                    {
                        "ego_dir": ego_frame_dir,
                        "exo_dir": exo_frame_dir,
                        "start_idx": f0,
                        "label": self.cls2idx[cls],
                    }
                )

    def __len__(self):
        return len(self.samples)

    def load_clip(self, frame_dir, start_idx, clip_len, transform):
        frames = []
        for idx in range(start_idx, start_idx + clip_len):
            offset = 1
            idx = idx + offset  # frame indices are 1-based
            frame_path = os.path.join(
                frame_dir, f"{os.path.basename(frame_dir)}-{idx:06d}.jpg"
            )
            # Safe load frame
            if not os.path.exists(frame_path):
                continue  # skip missing frame
            img = Image.open(frame_path).convert("RGB")
            frames.append(transform(img))

        if len(frames) < clip_len:  # missing some frames
            pad_len = clip_len - len(frames)
            frames += [frames[-1]] * pad_len  # repeat last frame

        return torch.stack(frames, 1)  # (C, T, H, W)

    def __getitem__(self, i):
        s = self.samples[i]
        ego_clip = self.load_clip(
            s["ego_dir"], s["start_idx"], self.clip_len, self.transform
        )
        exo_clip = self.load_clip(
            s["exo_dir"], s["start_idx"], self.clip_len, self.transform
        )

        return (ego_clip, exo_clip), s["label"]
