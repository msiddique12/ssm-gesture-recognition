import cv2
import torch
import numpy as np
import time
import pandas as pd
import os
import re
import socket
import threading
from collections import deque
from PIL import Image
from torchvision import transforms
from ssm_model import MambaGestureRecognizer

CHECKPOINT_PATH = "checkpoints/full_best_epoch15.pt" 
ANNOTATIONS_PATH = "annotations/jester-v1-validation.csv"
WINDOWS_IP = '172.29.208.1'
PORT = 65432

# Performance Settings
IMG_SIZE = 160
SEQ_LEN = 16
CONFIDENCE_THRESHOLD = 0.85
FRAMES_TO_LOCK = 2 
COOLDOWN_SECONDS = 1.2

class GestureController:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")
        
        # Load Model
        self.model = MambaGestureRecognizer(
            num_classes=27,
            seq_len=SEQ_LEN,
            img_size=IMG_SIZE,
            vit_name="vit_small_patch16_224",
            ssm_depth=2,
            ssm_state_dim=64,
            freeze_vit=False
        ).to(self.device)
        
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            else:
                self.model.load_state_dict(checkpoint)
            print("✅ Weights loaded.")
        else:
            print(f"❌ Checkpoint {CHECKPOINT_PATH} not found.")
            exit()

        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Logic State
        self.idx_to_label = self._load_labels()
        self.sock = self._connect_socket()
        
        # Threading Shared Variables
        self.lock = threading.Lock()
        self.running = False
        self.latest_frame_tensor = None
        self.display_status = "Initializing..."
        self.display_color = (255, 255, 255)
        
        # Logic Memory
        self.frame_buffer = deque(maxlen=SEQ_LEN)
        self.current_stable_gesture = "No gesture"
        self.consecutive_frames = 0
        self.last_raw_gesture = None
        self.waiting_for_reset = False
        self.last_volume_time = 0
        
        # NEW: Visual Feedback State
        self.last_trigger_time = 0
        self.last_trigger_name = ""

    def _load_labels(self):
        try:
            df = pd.read_csv(ANNOTATIONS_PATH, sep=';', header=None, names=["id", "label"])
            labels = sorted(df["label"].unique())
            return {i: name for i, name in enumerate(labels)}
        except:
            return {i: f"Class {i}" for i in range(27)}

    def _connect_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        try:
            s.connect((WINDOWS_IP, PORT))
            print(f"✅Connected to Windows at {WINDOWS_IP}")
            return s
        except:
            print("Running Offline (No Windows Connection)")
            return None

    def send_command(self, cmd):
        if self.sock:
            try:
                self.sock.sendall(cmd.encode('utf-8'))
            except:
                self.sock = None

    def inference_loop(self):
        """Runs in separate thread for speed."""
        print("🧠 AI Brain started (Latencies will print below).")
        
        while self.running:
            # 1. Grab latest frame safely
            tensor_input = None
            with self.lock:
                if self.latest_frame_tensor is not None:
                    tensor_input = self.latest_frame_tensor
            
            if tensor_input is None:
                time.sleep(0.01)
                continue

            # 2. Add to Rolling Buffer
            self.frame_buffer.append(tensor_input)
            
            # 3. Predict (Only if buffer is full)
            if len(self.frame_buffer) == SEQ_LEN:
                input_batch = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)
                
                # --- START LATENCY TIMER ---
                t0 = time.time()
                
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        logits = self.model(input_batch)
                        probs = torch.nn.functional.softmax(logits, dim=1)
                    conf, idx = torch.max(probs, 1)
                
                # Force GPU synchronization for accurate timing (Optional but recommended for benchmarking)
                torch.cuda.synchronize()
                
                # --- END LATENCY TIMER ---
                t1 = time.time()
                latency_ms = (t1 - t0) * 1000
                print(f"Inference Latency: {latency_ms:.2f} ms")
                
                idx = idx.item()
                conf = conf.item()
                
                # 4. Handle Logic
                self.update_logic(idx, conf)
            
            # Prevent 100% CPU usage
            time.sleep(0.01)

    def update_logic(self, idx, conf):
        # Filter Noise
        if conf > CONFIDENCE_THRESHOLD:
            raw_gesture = self.idx_to_label.get(idx, "Unknown")
        else:
            raw_gesture = "No gesture"

        # Stabilize
        if raw_gesture == self.last_raw_gesture:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = 0
            self.last_raw_gesture = raw_gesture

        # Trigger Actions
        if self.consecutive_frames >= FRAMES_TO_LOCK:
            if raw_gesture != self.current_stable_gesture:
                self.current_stable_gesture = raw_gesture
                
                # Reset Lock
                if raw_gesture == "No gesture":
                    self.waiting_for_reset = False
                
                # One-Shot Triggers
                elif not self.waiting_for_reset:
                    cmd = None
                    action_display = ""
                    
                    if raw_gesture == "Stop Sign": 
                        cmd = "PLAY_PAUSE"
                        action_display = "PLAY / PAUSE"
                    elif raw_gesture == "Swiping Left": 
                        cmd = "PREV"
                        action_display = "PREVIOUS VIDEO"
                    elif raw_gesture == "Swiping Right": 
                        cmd = "NEXT"
                        action_display = "NEXT VIDEO"
                    elif raw_gesture == "Sliding Two Fingers Down": 
                        cmd = "MINIMIZE"
                        action_display = "MINIMIZE WINDOW"
                    elif raw_gesture == "Zooming In With Full Hand": 
                        cmd = "FULLSCREEN"
                        action_display = "FULLSCREEN"
                    elif raw_gesture == "Zooming Out With Full Hand": 
                        cmd = "EXIT_FULLSCREEN"
                        action_display = "EXIT FULLSCREEN"
                    
                    if cmd:
                        self.send_command(cmd)
                        print(f"{cmd}")
                        self.waiting_for_reset = True
                        
                        # Set Visual Timer
                        self.last_trigger_time = time.time()
                        self.last_trigger_name = action_display

            # Continuous Triggers (Volume)
            current_time = time.time()
            if (current_time - self.last_volume_time) > 0.15:
                if self.current_stable_gesture == "Thumb Up":
                    self.send_command("VOL_UP")
                    self.last_volume_time = current_time
                    self.last_trigger_time = current_time
                    self.last_trigger_name = "VOLUME UP"
                elif self.current_stable_gesture == "Thumb Down":
                    self.send_command("VOL_DOWN")
                    self.last_volume_time = current_time
                    self.last_trigger_time = current_time
                    self.last_trigger_name = "VOLUME DOWN"

        # Update Display Text (Thread Safe)
        with self.lock:
            # PRIORITY 1: Show the Triggered Action (visual feedback for 1.0s)
            if (time.time() - self.last_trigger_time) < 1.0:
                self.display_status = f"ACTION: {self.last_trigger_name}"
                self.display_color = (0, 0, 255) # Red for Action
            
            # PRIORITY 2: Show Lock Status
            elif self.waiting_for_reset:
                self.display_status = "LOCKED (Drop Hand)"
                self.display_color = (0, 0, 255) 
            
            # PRIORITY 3: Show Ready Status
            elif self.current_stable_gesture != "No gesture":
                self.display_status = f"Ready: {self.current_stable_gesture}"
                self.display_color = (0, 255, 0) # Green
            
            # PRIORITY 4: Scanning
            else:
                self.display_status = "Scanning..."
                self.display_color = (0, 255, 255) # Yellow

    def run_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): cap = cv2.VideoCapture(1)
        
        # Force MJPG for Speed in WSL
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("❌ Camera failed. Run 'usbipd attach' in Windows.")
            return

        print("Camera Active. Press 'q' to quit.")
        
        # Start Inference Thread
        self.running = True
        t = threading.Thread(target=self.inference_loop)
        t.daemon = True
        t.start()

        fps_start = time.time()
        frame_cnt = 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Flip
            frame = cv2.flip(frame, 1)
            
            # 2. Send to Brain (Every 2nd frame)
            frame_cnt += 1
            if frame_cnt % 2 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tensor = self.transform(pil_img)
                
                with self.lock:
                    self.latest_frame_tensor = tensor

            # 3. Get Status
            status_text = ""
            status_color = (255,255,255)
            with self.lock:
                status_text = self.display_status
                status_color = self.display_color

            # 4. Draw UI
            now = time.time()
            fps = 1.0 / (now - fps_start)
            fps_start = now
            
            # Draw Background Box for text
            cv2.rectangle(frame, (0,0), (640, 60), (0,0,0), -1)
            
            # FPS
            cv2.putText(frame, f"FPS: {fps:.0f}", (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)
            
            # Status Text
            cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
            
            cv2.imshow('Mamba Gesture Controller', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        if self.sock: self.sock.close()

if __name__ == "__main__":
    controller = GestureController()
    controller.run_camera()