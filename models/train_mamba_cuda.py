import os
import sys
import time
import random
import platform

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from jester_dataset import JesterVideoDataset
from ssm_model import MambaGestureRecognizer

torch.set_num_threads(4)

#enable TF32 for faster matmul on Ampere GPUs (RTX 30 series)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(
    batch_size: int = 16,  
    seq_len: int = 16,
    img_size: int = 160,
    quick_test: bool = False,
):
    base_ann = os.path.join(CURRENT_DIR, "annotations")

    # 2. Videos are in ~/jester_data/20bn-jester-v1
    user_home = os.path.expanduser("~")
    data_root_path = os.path.join(user_home, "jester_data", "20bn-jester-v1")

    print(f"[Data] Looking for videos in: {data_root_path}")
    print(f"[Data] Looking for annotations in: {base_ann}")

    if quick_test:
        train_csv = os.path.join(base_ann, "jester-v1-train-quick-testing.csv")
        val_csv   = os.path.join(base_ann, "jester-v1-validation-quick-testing.csv")
    else:
        train_csv = os.path.join(base_ann, "jester-v1-train.csv")
        val_csv   = os.path.join(base_ann, "jester-v1-validation.csv")

    train_ds = JesterVideoDataset(
        data_root=data_root_path,
        annotations_file=train_csv,
        seq_len=seq_len,
        img_size=img_size,
        is_train=True,
    )

    val_ds = JesterVideoDataset(
        data_root=data_root_path,
        annotations_file=val_csv,
        seq_len=seq_len,
        img_size=img_size,
        is_train=False,
    )

    print("Train samples:", len(train_ds), " Val samples:", len(val_ds))
    print("Num classes:", len(train_ds.label_to_idx))

    # WORKER SAFETY: Set to 2 to prevent WSL Bus Errors
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Safer than 4 on WSL
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2, 
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2, # Safer than 4
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader, len(train_ds.label_to_idx)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, total_epochs, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    total = 0
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)

    for step, (videos, labels) in enumerate(loader, start=1):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixed precision
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(videos)
            loss = criterion(logits, labels)
            loss_value = loss.item()
            loss = loss / accumulation_steps 

        scaler.scale(loss).backward()

        if step % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss_value * labels.size(0)
        total += labels.size(0)

        # Log every 50 steps
        if step % 50 == 0: 
            avg_loss = running_loss / max(total, 1)
            elapsed = time.time() - start_time
            samples_per_sec = total / elapsed if elapsed > 0 else 0.0
            eta_seconds = (len(loader) - step) / (step / elapsed)
            eta_mins = eta_seconds / 60.0
            
            print(
                f"  [Ep {epoch} | step {step}/{len(loader)}] "
                f"loss={loss_value:.4f} (avg {avg_loss:.4f}) | "
                f"{samples_per_sec:.1f} img/s | ETA: {eta_mins:.1f}m"
            )

    return running_loss / max(total, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        for videos, labels in loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(videos)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device:", device, torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA not available.")
        return

    # --- FULL FINE-TUNING CONFIG ---
    img_size = 160 
    seq_len = 16
    
    batch_size = 16          
    accumulation_steps = 2  # Effective batch size = 32 (16 * 2)
    
    #slightly higher LR than hybrid because we have more params to move
    lr = 2e-5               
    
    num_epochs = 15         
    
    print(f"Full Tuning Mode: LR={lr}, Batch={batch_size}, Accum={accumulation_steps}")

    train_loader, val_loader, num_classes = make_dataloaders(
        batch_size=batch_size,
        seq_len=seq_len,
        img_size=img_size,
        quick_test=False,
    )

    model = MambaGestureRecognizer(
        num_classes=num_classes,
        seq_len=seq_len,
        img_size=img_size,
        vit_name="vit_small_patch16_224",
        ssm_depth=2,
        ssm_state_dim=64,
        freeze_vit=False
    ).to(device)
    
    # Load the best model from the HYBRID run
    resume_path = "checkpoints/hybrid_best_epoch6.pt"
    
    if os.path.exists(resume_path):
        print(f"\n📥 Loading model from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            best_val_acc = checkpoint['val_acc']
        else:
            model.load_state_dict(checkpoint)
            best_val_acc = 0.654 # Approximate
            
        print(f"✅ Loaded. Starting Full Tuning from {best_val_acc*100:.2f}% base.\n")
    else:
        print(f"⚠️ Warning: Checkpoint {resume_path} not found! Starting fresh.")
        best_val_acc = 0.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    os.makedirs("checkpoints", exist_ok=True)

    print(f"\n{'='*50}\nStarting Full Fine-Tuning\n{'='*50}")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, num_epochs, accumulation_steps
        )
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch} Done in {epoch_time/60:.1f}m | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
        )
        
        torch.cuda.reset_peak_memory_stats()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"checkpoints/full_best_epoch{epoch}.pt")
            print(f"✅ NEW RECORD! Saved model ({val_acc*100:.2f}%)")

if __name__ == "__main__":
    main()