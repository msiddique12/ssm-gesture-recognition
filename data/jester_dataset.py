import os
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_frame_transform(is_train: bool, img_size: int = 224):
    resize_size = int(img_size * 1.15)
    if is_train:
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomCrop(img_size),
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
    if num_frames <= 0: return [0] * seq_len
    if num_frames >= seq_len:
        return torch.linspace(0, num_frames - 1, steps=seq_len).long().tolist()
    else:
        base = list(range(num_frames))
        while len(base) < seq_len:
            base.append(num_frames - 1)
        return base[:seq_len]

class JesterVideoDataset(Dataset):
    def __init__(self, data_root: str, annotations_file: str, seq_len: int = 16, img_size: int = 224, is_train: bool = True):
        super().__init__()
        self.data_root = data_root
        self.frames_root = data_root 
        self.seq_len = seq_len
        self.img_size = img_size
        self.transform = build_frame_transform(is_train, img_size)

        self.df = pd.read_csv(
            annotations_file, 
            sep=';', 
            header=None, 
            names=["video_id", "label"],
            dtype={"video_id": str} 
        )

        label_names = sorted(self.df["label"].unique())
        self.label_to_idx = {name: i for i, name in enumerate(label_names)}
        self.df["label_idx"] = self.df["label"].map(self.label_to_idx)

    def __len__(self) -> int:
        return len(self.df)

    def _get_frame_paths(self, video_id: str) -> List[str]:
        folder = os.path.join(self.frames_root, str(video_id))
        
        if not os.path.isdir(folder):
            return [] 

        frame_files = sorted(f for f in os.listdir(folder) if f.endswith('.jpg'))
        return [os.path.join(folder, f) for f in frame_files]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        video_id = str(row["video_id"])
        label_idx = int(row["label_idx"])

        frame_paths = self._get_frame_paths(video_id)
        
        if not frame_paths:
            return torch.zeros(self.seq_len, 3, self.img_size, self.img_size), label_idx

        indices = sample_indices(len(frame_paths), self.seq_len)
        frames = []
        for i in indices:
            try:
                img = Image.open(frame_paths[i]).convert("RGB")
                frames.append(self.transform(img))
            except Exception:
                frames.append(torch.zeros(3, self.img_size, self.img_size))
        
        return torch.stack(frames), label_idx