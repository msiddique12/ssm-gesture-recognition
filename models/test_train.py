import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data.jester_dataset import JesterVideoDataset
from models.ssm_model import MambaGestureRecognizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = JesterVideoDataset(
        data_root="data/jester_small",
        annotations_file="data/jester_small/annotations.csv",
        seq_len=16,
        img_size=224,
        is_train=True,
    )
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)

    model = MambaGestureRecognizer(
        num_classes=len(train_ds.label_to_idx),
        seq_len=16,
        img_size=224,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for videos, labels in train_loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(videos)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        print("Loss:", float(loss))
        break  # just a smoke test

if __name__ == "__main__":
    main()
