import cv2
import os
import subprocess
import time
import sys  # Added to get the current Python executable
import torch

def capture_frames(output_dir='live_input', frame_limit=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(0)
    index = 0

    print("[INFO] Starting frame capture. Press 'q' to quit.")
    while index < frame_limit:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        filename = os.path.join(output_dir, f"{index:06d}.png")
        cv2.imwrite(filename, frame)
        index += 1

        cv2.imshow("Live Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Capture terminated by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captured {index} frames.")

def run_slam3r(input_dir='live_input', output_dir='recon_output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("[INFO] Running SLAM3R reconstruction...")

    recon_script = os.path.join("recon.py")

    if not os.path.isfile(recon_script):
        raise FileNotFoundError(f"Could not find: {recon_script}")

    subprocess.run([
        sys.executable, recon_script,
        "--img_dir", "live_input",
        "--output_dir", "recon_output",
        # "--ckpt_path", "SLAM3R/checkpoints/slam3r.pth",  # adjust path if needed
        "--save_mesh",
        "--vis"
    ], check=True)
    print("[INFO] Reconstruction complete.")

if __name__ == "__main__":
    capture_frames()
    run_slam3r()
