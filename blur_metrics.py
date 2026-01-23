# blur_metrics.py
import numpy as np
import cv2

def _safe_gray_from_exr(exr3: np.ndarray) -> np.ndarray:
    """
    exr3: float16/float32, shape (H,W,3) in BGR or RGB (we assume BGR like OpenCV).
    Return gray float32 (H,W) in linear domain, with NaN/Inf cleaned.
    """
    img = np.nan_to_num(exr3, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if img.ndim == 2:
        gray = img
    else:
        # OpenCV imread gives BGR by default
        b = img[..., 0]
        g = img[..., 1]
        r = img[..., 2]
        # Rec.709 luma
        gray = 0.0722 * b + 0.7152 * g + 0.2126 * r

    # Remove negatives (linear EXR can contain small negatives)
    gray = np.maximum(gray, 0.0)
    return gray