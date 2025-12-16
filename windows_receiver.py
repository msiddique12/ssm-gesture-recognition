import socket
import pyautogui

# Config
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 65432      # Communication port

def start_server():
    print(f"🦻 Windows Receiver Listening on port {PORT}...")
    print("   (Click on your YouTube window to make sure it has focus!)")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"✅ Waiting for connection from WSL...")
        
        conn, addr = s.accept()
        with conn:
            print(f"🔗 Connected by {addr}")
            while True:
                try:
                    data = conn.recv(1024)
                    if not data:
                        print("Connection closed. Waiting for new connection...")
                        conn, addr = s.accept()
                        continue
                    
                    command = data.decode('utf-8').strip()
                    print(f"   >>> Received: {command}")
                    
                    # --- YOUTUBE OPTIMIZED MAPPING ---
                    if command == "PLAY_PAUSE":
                        # 'k' is the dedicated Play/Pause shortcut for YouTube
                        pyautogui.press('k') 
                        
                    elif command == "VOL_UP":
                        pyautogui.press('volumeup') # Global volume is fine
                        
                    elif command == "VOL_DOWN":
                        pyautogui.press('volumedown') # Global volume is fine
                        
                    elif command == "PREV":
                        # YouTube Shortcut for Previous Video
                        pyautogui.hotkey('shift', 'p')
                        
                    elif command == "NEXT":
                        # YouTube Shortcut for Next Video
                        pyautogui.hotkey('shift', 'n')
                    
                    elif command == "FULLSCREEN":
                        pyautogui.press('f') # YouTube shortcut
                        
                    elif command == "EXIT_FULLSCREEN":
                        pyautogui.press('f') # 'f' toggles it, or use 'escape'
                        
                except Exception as e:
                    print(f"Error: {e}")
                    break

if __name__ == "__main__":
    pyautogui.FAILSAFE = True
    start_server()