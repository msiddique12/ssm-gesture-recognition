# Real-Time Gesture Control with State Space Models (Mamba)

A real-time hand gesture recognition system that controls Windows applications (YouTube, Media Player) using a webcam. This project introduces a novel Hybrid Architecture combining a Vision Transformer (ViT) for spatial feature extraction and a Mamba State Space Model (SSM) for temporal modeling.

Because the Mamba kernels require Linux/CUDA but the target applications (Media Player, Browser) run on Windows, this system uses a Split-Architecture:

1. The "Brain" (WSL 2 / Linux): Runs the Deep Learning model. It captures the webcam feed, processes frames, and predicts gestures. It sends commands via TCP/IP.
2. The "Hands" (Windows Host): A lightweight Python script (windows_receiver.py) that listens for TCP commands and simulates physical key presses.

Performance:
Latency: ~16ms per inference (60+ FPS throughput).
Accuracy: 72.0% on Jester Dataset (27 Classes).
Hardware: Optimized for NVIDIA RTX 30-series GPUs.

Setup & Installation

1. Windows Host (The Receiver)
This script runs on your main Windows OS to control the mouse/keyboard.

pip install pyautogui

2. WSL 2 / Linux (The Brain)
This runs the heavy AI model. You must have CUDA installed.

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Mamba SSM (Requires NVCC compiler)
pip install mamba-ssm causal-conv1d

# Install dependencies
pip install -r requirements.txt

Usage Guide

Step 1: Start the Receiver (Windows)
Open PowerShell in the project folder:

python windows_receiver.py

Status: "Listening on 0.0.0.0:65432..."

Step 2: Start the AI (WSL/Linux)
Open your WSL terminal:

python inference_linux.py

Note: The script automatically detects your Windows Host IP via /etc/resolv.conf.

Step 3: Control!
The webcam window will appear. Perform gestures to control your PC.

Supported Gestures
Stop Sign: Play / Pause
Swiping Left: Previous Video
Swiping Right: Next Video
Thumb Up: Volume Up
Thumb Down: Volume Down
Sliding Two Fingers Down: Minimize Window
Zooming In (Full Hand): Fullscreen

Training
To retrain the model on the Jester dataset:

python train_mamba_cuda.py

The training uses a progressive strategy (Frozen ViT -> Hybrid -> Full Fine-Tuning) to achieve 72% accuracy.
