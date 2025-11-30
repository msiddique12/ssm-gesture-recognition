import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms


# ImageNet stats – good defaults if we later plug in a Vision Transformer
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_frame_transform(is_train: bool, img_size: int = 224):
    """
    Basic spatial transforms for each frame.
    Kept simple for now; we can add more augmentation later.
    """
    resize_size = int(img_size * 1.15)

    if is_train:
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomCrop(img_size),
            # NOTE: We skip RandomHorizontalFlip for now because Jester has
            # left/right-specific gestures (Swiping Left vs Right).
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def sample_indices(num_frames: int, seq_len: int) -> List[int]:
    """
    Uniformly sample `seq_len` frame indices from a video with `num_frames` frames.

    For now: simple linspace-style sampling. If the video is shorter than `seq_len`,
    we repeat the last frame until we have enough.
    """
    if num_frames <= 0:
        raise ValueError("Video has no frames.")

    if num_frames >= seq_len:
        # evenly spaced indices from 0 .. num_frames-1
        indices = torch.linspace(0, num_frames - 1, steps=seq_len).long().tolist()
    else:
        # take all frames, then repeat last one
        base = list(range(num_frames))
        while len(base) < seq_len:
            base.append(num_frames - 1)
        indices = base[:seq_len]

    return indices


class JesterVideoDataset(Dataset):
    """
    Loads Jester-style videos (as folders of frames) and returns
    a tensor of shape [T, 3, H, W] plus an integer label.

    We assume a directory structure like:

        data_root/
          20bn-jester-v1/
            1/
              00001.jpg
              00002.jpg
              ...
            2/
            ...
          annotations.csv  (video_id,label)

    This class just handles *preprocessing* and organization; the model
    and training loop will live elsewhere.
    """

    def __init__(
        self,
        data_root: str,
        annotations_file: str,
        seq_len: int = 16,
        img_size: int = 224,
        is_train: bool = True,
    ):
        super().__init__()

        self.data_root = data_root
        self.frames_root = os.path.join(data_root, "20bn-jester-v1")
        self.seq_len = seq_len

        # Read CSV with columns ["video_id", "label"]
        self.df = pd.read_csv(annotations_file)

        # Build label_name -> index mapping (sorted for consistency)
        label_names = sorted(self.df["label"].unique())
        self.label_to_idx = {name: i for i, name in enumerate(label_names)}
        self.idx_to_label = {i: name for name, i in self.label_to_idx.items()}

        # Map labels to indices once so __getitem__ is fast
        self.df["label_idx"] = self.df["label"].map(self.label_to_idx)

        # Transform for individual frames
        self.transform = build_frame_transform(is_train=is_train, img_size=img_size)

    def __len__(self) -> int:
        # Total number of video clips
        return len(self.df)

    def _get_frame_paths(self, video_id: int) -> List[str]:
        """
        Given a video id, return all frame paths for that clip, sorted.
        """
        folder = os.path.join(self.frames_root, str(video_id))
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Video folder not found: {folder}")

        frame_files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not frame_files:
            raise RuntimeError(f"No frames found in {folder}")

        return [os.path.join(folder, f) for f in frame_files]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Fetch a single video clip and its label.

        Returns:
            video_tensor: [T, 3, H, W]
            label_idx:   int in [0, num_classes)
        """
        row = self.df.iloc[idx]
        video_id = int(row["video_id"])
        label_idx = int(row["label_idx"])

        # Collect all frame file paths for this video
        frame_paths = self._get_frame_paths(video_id)
        num_frames = len(frame_paths)

        # Decide which frames to use
        indices = sample_indices(num_frames, self.seq_len)

        frames = []
        for i in indices:
            img_path = frame_paths[i]
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            frames.append(img)

        # Stack to shape [T, 3, H, W]
        video_tensor = torch.stack(frames, dim=0)

        return video_tensor, label_idx
