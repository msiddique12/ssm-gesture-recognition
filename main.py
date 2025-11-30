import torch
from torch.utils.data import DataLoader
from data.jester_dataset import JesterVideoDataset

def main():
    dataset = JesterVideoDataset(
        data_root="data/jester_small",
        annotations_file="data/jester_small/annotations.csv",
        seq_len=16,
        img_size=224,
        is_train=True,
    )

    print("Num samples:", len(dataset))
    print("Num classes:", len(dataset.label_to_idx))

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    videos, labels = next(iter(loader))
    print("Batch video shape:", videos.shape)   # [B, T, 3, H, W]
    print("Batch labels:", labels)

if __name__ == "__main__":
    main()
