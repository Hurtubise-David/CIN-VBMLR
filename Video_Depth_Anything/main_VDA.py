import numpy as np
import torch
import cv2
import time
import os

from video_depth_anything.video_depth_stream import VideoDepthAnything
from utils.dc_utils import apply_depth_colormap

# ==== Configuration fixe ====
CAMERA_ID = 0                 # Modifier si ta caméra ELP est à un autre index
INPUT_SIZE = 518              # Taille d’entrée du modèle (crop carré)
MAX_RES = 1280                # Redimensionne si plus grand (sinon laisse comme tel)
ENCODER = 'vitl'              # Fixé à ViT-Large
USE_FP32 = False              # False = float16 (rapide)
GRAYSCALE = False             # True = pas de couleurs dans la visualisation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"[INFO] Using device: {DEVICE}")

# ==== Chargement du modèle ====
model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

print("[INFO] Loading model weights...")
video_depth_anything = VideoDepthAnything(**model_configs[ENCODER])
checkpoint = f'./checkpoints/video_depth_anything_{ENCODER}.pth'
video_depth_anything.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
video_depth_anything = video_depth_anything.to(DEVICE).eval()
print("[INFO] Model ready!")

# ==== Ouverture de la caméra ====
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"[ERROR] Cannot open camera with ID {CAMERA_ID}")
    exit()

print("[INFO] Streaming from camera. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to grab frame.")
        break

    # Traitement de l’image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    if MAX_RES > 0 and max(h, w) > MAX_RES:
        scale = MAX_RES / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # Inférence profondeur
    with torch.no_grad():
        depth = video_depth_anything.infer_video_depth_one(
            frame, input_size=INPUT_SIZE, device=DEVICE, fp32=USE_FP32
        )

    # Visualisation
    depth_vis = apply_depth_colormap(depth, grayscale=GRAYSCALE)
    cv2.imshow("Depth (Press q to quit)", depth_vis)

    # Sortie
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
