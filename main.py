#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dependancies
    pip install PyQt5 opencv-python numpy pyyaml psutil pyqtgraph PyOpenGL
"""
import os
import sys
import json
import csv
import time
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2

import numpy as np
import psutil
import yaml
import glob
import re
import torch

from PyQt5 import QtCore, QtGui, QtWidgets

# --- VDA paths ---
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "Video_Depth_Anything", "video_depth_anything"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Video_Depth_Anything"))

from video_depth_stream import VideoDepthAnything as VDAStream

# --- TBD: Live pose module ---
try:
    from live_pose_page import LivePosePage
except Exception:
    # Fallback placeholder if module is not available
    class LivePosePage(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            lay = QtWidgets.QVBoxLayout(self)
            lbl = QtWidgets.QLabel("<h2>Live Pose</h2><div>Module manquant (live_pose_page.py)</div>")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lay.addStretch(1); lay.addWidget(lbl); lay.addStretch(1)

# --- Branding / Logo ---
APP_NAME = "CIN-VBMLR"
LOGO_CANDIDATES = ["CIN_logo.png"]

SCRIPT_DIR = Path(__file__).resolve().parent

def find_logo_path() -> Path:
    for name in LOGO_CANDIDATES:
        p = SCRIPT_DIR / name
        if p.exists():
            return p
    return None

def exr_dump_header(exr_path: str, max_keys: int = 80):
    try:
        import OpenEXR, Imath
        exr = OpenEXR.InputFile(str(exr_path))
        hdr = exr.header()
        keys = sorted(list(hdr.keys()))
        print("---- EXR HEADER KEYS ----")
        for k in keys[:max_keys]:
            v = hdr[k]
            t = type(v).__name__
            s = str(v)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"{k}  [{t}]  {s}")
        if len(keys) > max_keys:
            print(f"... (+{len(keys)-max_keys} autres)")
        print("--------------------------")
        exr.close()
    except Exception as e:
        print(f"[EXR] header dump error: {e}")

# --- 3D viewer (OpenGL) ---
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    HAS_PG_GL = True
except Exception:
    HAS_PG_GL = False


class OptiTrack3DViewer(QtWidgets.QWidget):

    def __init__(self, parent=None, max_points=5000, mm_to_m=True):
        super().__init__(parent)

        self.max_points = int(max(100, max_points))
        self.mm_to_m = bool(mm_to_m)

        self._origin = None  # recentrer au 1er sample
        self._traj = deque(maxlen=self.max_points)

        # pose courante (m, recentrée)
        self._cur_p = None
        self._cur_q = None  # (x,y,z,w)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        if not HAS_PG_GL:
            lbl = QtWidgets.QLabel("pyqtgraph.opengl non disponible.\nInstalle: pip install pyqtgraph PyOpenGL")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet("color:#aaa; background:#111; border:1px solid #333;")
            lay.addWidget(lbl)
            return

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((10, 10, 10, 255))
        lay.addWidget(self.view, 1)

        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.25, 0.25)
        grid.translate(0, 0, 0)
        self.view.addItem(grid)

        self.axis = gl.GLAxisItem()
        self.axis.setSize(1.0, 1.0, 1.0)
        self.view.addItem(self.axis)

        self.traj_item = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            width=2.0, antialias=True, mode='line_strip'
        )
        self.view.addItem(self.traj_item)

        self.pt_item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            size=10.0
        )
        self.view.addItem(self.pt_item)

        # --- frustum items (2 items: rays + rectangle) ---
        self.frustum_rays = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            width=1.0, antialias=True, mode='lines'
        )
        self.frustum_box = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            width=1.0, antialias=True, mode='line_strip'
        )
        self.view.addItem(self.frustum_rays)
        self.view.addItem(self.frustum_box)
        self._hide_frustum()

        self.view.setCameraPosition(distance=3.5, elevation=18, azimuth=45)

        # Controls
        bar = QtWidgets.QHBoxLayout()
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.chk_follow = QtWidgets.QCheckBox("Follow"); self.chk_follow.setChecked(True)

        # frustum toggle + scale
        self.chk_frustum = QtWidgets.QCheckBox("Frustum"); self.chk_frustum.setChecked(False)
        self.frustum_scale = QtWidgets.QDoubleSpinBox()
        self.frustum_scale.setRange(0.05, 10.0)
        self.frustum_scale.setSingleStep(0.05)
        self.frustum_scale.setDecimals(2)
        self.frustum_scale.setValue(1.0)

        bar.addWidget(self.btn_clear)
        bar.addWidget(self.chk_follow)
        bar.addSpacing(12)
        bar.addWidget(self.chk_frustum)
        bar.addWidget(QtWidgets.QLabel("Scale"))
        bar.addWidget(self.frustum_scale)
        bar.addStretch(1)
        lay.addLayout(bar)

        self.btn_clear.clicked.connect(self.clear)
        self.chk_frustum.toggled.connect(lambda _: self._update_frustum())
        self.frustum_scale.valueChanged.connect(lambda _: self._update_frustum())

    # -------------------- Helpers -------------------- #
    def _hide_frustum(self):
        self.frustum_rays.setData(pos=np.zeros((1, 3), dtype=np.float32))
        self.frustum_box.setData(pos=np.zeros((1, 3), dtype=np.float32))

    def clear(self):
        self._traj.clear()
        self._origin = None
        self._cur_p = None
        self._cur_q = None
        if HAS_PG_GL:
            self.traj_item.setData(pos=np.zeros((1, 3), dtype=np.float32))
            self.pt_item.setData(pos=np.zeros((1, 3), dtype=np.float32))
            self._hide_frustum()

    def _mm_to_units(self, px, py, pz):
        if self.mm_to_m:
            return np.array([px, py, pz], dtype=np.float32) / 1000.0
        return np.array([px, py, pz], dtype=np.float32)

    def _apply_origin(self, p: np.ndarray):
        if self._origin is None:
            self._origin = p.copy()
        return p - self._origin

    def _quat_to_R(self, qx, qy, qz, qw):
        q = np.array([qx, qy, qz, qw], dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n < 1e-9:
            return np.eye(3, dtype=np.float32)
        q /= n
        x, y, z, w = q
        # Rotation matrix (x,y,z,w)
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float32)
        return R

    # -------------------- Trajectory -------------------- #
    def set_current_pose_mm(self, px, py, pz, qx=None, qy=None, qz=None, qw=None):
        """Pose OptiTrack -> update point courant + frustum (si actif)."""
        p = self._mm_to_units(px, py, pz)
        p = self._apply_origin(p)

        self._cur_p = p
        self._cur_q = (qx, qy, qz, qw) if (qx is not None and qy is not None and qz is not None and qw is not None) else None

        self.pt_item.setData(pos=p.reshape(1, 3))

        if self.chk_follow.isChecked():
            self.view.opts['center'] = pg.Vector(p[0], p[1], p[2])
            self.view.update()

        self._update_frustum()

    def set_trajectory_mm(self, points_mm, reset_origin=True):
        """
        points_mm: iterate from (x_mm,y_mm,z_mm).
        """
        pts = []
        for pmm in points_mm:
            if pmm is None:
                continue
            px, py, pz = pmm
            p = self._mm_to_units(px, py, pz)
            pts.append(p)

        if reset_origin:
            self._origin = None

        if not pts:
            self._traj.clear()
            self.traj_item.setData(pos=np.zeros((1, 3), dtype=np.float32))
            return

        # apply origin consistently
        pts2 = []
        for p in pts:
            p2 = self._apply_origin(p)
            pts2.append(p2)

        # cap
        if len(pts2) > self.max_points:
            pts2 = pts2[-self.max_points:]

        arr = np.asarray(pts2, dtype=np.float32)
        if arr.shape[0] < 2:
            arr = np.vstack([arr, arr])

        self.traj_item.setData(pos=arr)

    # -------------------- Frustum -------------------- #
    def _update_frustum(self):
        if not HAS_PG_GL:
            return
        if not self.chk_frustum.isChecked():
            self._hide_frustum()
            return
        if self._cur_p is None:
            self._hide_frustum()
            return

        p = self._cur_p
        q = self._cur_q

        # frustum local (camera frame): origin + rectangle at depth
        s = float(self.frustum_scale.value())
        depth = 0.25 * s
        hw = 0.10 * s
        hh = 0.075 * s

        O = np.array([0, 0, 0], dtype=np.float32)
        c1 = np.array([-hw, -hh, depth], dtype=np.float32)
        c2 = np.array([ hw, -hh, depth], dtype=np.float32)
        c3 = np.array([ hw,  hh, depth], dtype=np.float32)
        c4 = np.array([-hw,  hh, depth], dtype=np.float32)

        corners = np.stack([c1, c2, c3, c4], axis=0)

        if q is not None:
            qx, qy, qz, qw = q
            R = self._quat_to_R(qx, qy, qz, qw)
            corners_w = (R @ corners.T).T
            O_w = (R @ O.reshape(3, 1)).reshape(3)
        else:
            corners_w = corners
            O_w = O

        # translate
        corners_w = corners_w + p.reshape(1, 3)
        O_w = O_w + p

        # rays: (O->c1, O->c2, O->c3, O->c4) as "lines" = pairs
        rays = np.vstack([
            O_w, corners_w[0],
            O_w, corners_w[1],
            O_w, corners_w[2],
            O_w, corners_w[3],
        ]).astype(np.float32)

        # box: c1->c2->c3->c4->c1
        box = np.vstack([corners_w, corners_w[0]]).astype(np.float32)

        self.frustum_rays.setData(pos=rays)
        self.frustum_box.setData(pos=box)



# ============================ Utils ============================ #
FOURCC_CANDIDATES = [
    ("mp4v", ".mp4"),   # Windows compatibility
    ("MJPG", ".avi"),   
    ("XVID", ".avi"),
    ("X264", ".mkv"),
]
WARMUP_GRABS = 10
QUEUE_MAX = 180  # ~3s @60fps

def exr_read_timecode(exr_path: str):
    """
    Return SMPTE TC (HH:MM:SS:FF) if found in header EXR.
    attribute EXR (Imath.TimeCode or string timecode).
    """
    try:
        import OpenEXR, Imath
        exr = OpenEXR.InputFile(str(exr_path))
        hdr = exr.header()

        # Keys
        candidate_keys = [
            "timeCode", "timecode",
            "smpte:timecode", "smpte:timeCode",
            "com.ARRI.timecode", "arri:timecode",
            "utc:timecode", "tc", "Timecode", "TimeCode",
        ]

        def is_smpte(s: str) -> bool:
            # Format HH:MM:SS:FF
            if not s:
                return False
            s = s.strip().replace('"', '')
            parts = s.split(":")
            if len(parts) != 4:
                return False
            try:
                hh, mm, ss, ff = [int(p) for p in parts]
                return (0 <= hh <= 23) and (0 <= mm <= 59) and (0 <= ss <= 59) and (0 <= ff <= 299)
            except Exception:
                return False

        # 1) Search key
        for k in candidate_keys:
            if k in hdr:
                v = hdr[k]

                # TimeCode Type
                if hasattr(v, "hours") and hasattr(v, "frame"):
                    tc = f"{int(v.hours):02d}:{int(v.minutes):02d}:{int(v.seconds):02d}:{int(v.frame):02d}"
                    exr.close()
                    return tc

                # Else string
                s = str(v).strip().replace('"', '')
                if is_smpte(s):
                    exr.close()
                    return s

        # 2) keys with "timecode" or finishing by "tc"
        for k in hdr.keys():
            kl = k.lower()
            if ("timecode" in kl) or kl.endswith("tc") or ("_tc" in kl):
                v = hdr[k]
                if hasattr(v, "hours") and hasattr(v, "frame"):
                    tc = f"{int(v.hours):02d}:{int(v.minutes):02d}:{int(v.seconds):02d}:{int(v.frame):02d}"
                    exr.close()
                    return tc
                s = str(v).strip().replace('"', '')
                if is_smpte(s):
                    exr.close()
                    return s

        exr.close()
        return None
    except Exception:
        return None

def build_vda_wrapper():
    # (encoder, fp16, device)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vda = VideoDepthAnything(
        encoder="vits",        # or "vitl"
        device=device,
        fp16=(device == "cuda")
    )
    return vda

def depth_to_vis_u8(depth: np.ndarray):
    """depth float32 -> BGR uint8 for show."""
    if depth is None:
        return None
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # normalisation
    lo = np.percentile(d, 2.0)
    hi = np.percentile(d, 98.0)
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    dn = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    u8 = (dn * 255.0).astype(np.uint8)
    # colormap
    vis = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return vis

def exr_float_to_vda_bgr8(exr3: np.ndarray, gamma=2.2):
    """
    Convert EXR linear float (BGR/RGB) -> BGR uint8 for VDA.
    """
    bgr8 = exr_to_preview_bgr8(exr3, gamma=gamma) 
    return bgr8

def frame_num_from_exr_filename(exr_path: str) -> int:
    """
    Ex: A002C001_250813_C5HH.00000001.exr -> 1
    Return -1 if not available.
    """
    name = Path(exr_path).name
    m = re.search(r"\.(\d+)\.exr$", name, flags=re.IGNORECASE)
    if not m:
        return -1
    return int(m.group(1))


def natural_key(path: str):
    # Search: ..._2.exr before ..._10.exr
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(path).name)]

def exr_to_preview_bgr8(exr_rgb_or_bgr: np.ndarray, gamma: float = 2.2):
    """
    exr_* in float32/float16 (linear), 3 canals.
    Return BGR image uint8 to screen.
    NOTE: "view transform" for UI.
    """
    img = exr_rgb_or_bgr
    if img is None:
        return None
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)

    # clamp for preview
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.maximum(img, 0.0)

    # simple tonemap (Reinhard)
    img = img / (1.0 + img)

    # gamma display
    img = np.power(img, 1.0 / gamma)

    # 0..255
    img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    
    return img8



def now_utc_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def now_local_iso():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class Timecode:
    def __init__(self, fps: float = 60.0):
        self.set_fps(fps)

    def set_fps(self, fps: float):
        self.fps = max(1.0, float(fps))

    def from_timestamp(self, ts: float) -> str:
        dt = datetime.fromtimestamp(ts).astimezone()
        ff = int(((ts - int(ts)) * self.fps) + 1e-6)
        return f"{dt:%H:%M:%S}:{ff:02d}"


# ============================ Writer Worker ============================ #
class VideoWriterWorker(QtCore.QObject):
    status_msg = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.queue = deque(maxlen=QUEUE_MAX)
        self._stop = False
        self.writer = None
        self.csv_file = None
        self.csv_writer = None
        self.opened = False
        self.video_path = None
        self.fourcc_used = None
        self._written_frames = 0
        self._dropped_count = 0
        self.expected_wh = None

    def open(self, video_base: Path, w: int, h: int, fps: float):
        for fourcc_str, ext in FOURCC_CANDIDATES:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            path = str(video_base) + ext
            vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if vw.isOpened():
                self.writer = vw
                self.video_path = Path(path)
                self.fourcc_used = fourcc_str
                self.opened = True
                self.expected_wh = (w, h)
                return True
        return False

    @QtCore.pyqtSlot()
    def stop(self):
        self._stop = True

    def close(self):
        if self.writer is not None:
            self.writer.release(); self.writer = None
        if self.csv_file is not None:
            self.csv_file.close(); self.csv_file = None; self.csv_writer = None
        self.opened = False

    def push(self, item):
        if len(self.queue) >= self.queue.maxlen:
            self.queue.popleft(); self._dropped_count += 1
        self.queue.append(item)

    def loop(self):
        while not self._stop or len(self.queue) > 0:
            if not self.queue:
                time.sleep(0.002)
                continue
            frame, meta = self.queue.popleft()
            if self.writer is not None:
                try:
                    if frame is None:
                        continue
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    if (frame.shape[1], frame.shape[0]) != self.expected_wh:
                        frame = cv2.resize(frame, self.expected_wh, interpolation=cv2.INTER_AREA)
                    frame = np.ascontiguousarray(frame)
                    self.writer.write(frame)
                    self._written_frames += 1
                except Exception as e:
                    self.status_msg.emit(f"Writer: exception OpenCV — frame skipped ({e}).")
                    continue
        self.status_msg.emit(f"Writer: finish ({self._written_frames} frames, cumul drops: {self._dropped_count}).")


# ============================ Capture Worker ============================ #
class VDAWorker(QtCore.QObject):
    depth_ready = QtCore.pyqtSignal(np.ndarray, float)  # depth, timestamp
    status_msg = QtCore.pyqtSignal(str)

    def __init__(self, vda_wrapper):
        super().__init__()
        self.vda = vda_wrapper
        self._lock = threading.Lock()
        self._latest = None
        self._stop = False

    @QtCore.pyqtSlot(np.ndarray, float)
    def submit(self, frame_bgr, ts):
        with self._lock:
            self._latest = (frame_bgr.copy(), ts)

    @QtCore.pyqtSlot()
    def run_loop(self):
        self.status_msg.emit("VDA: worker start")
        while not self._stop:
            item = None
            with self._lock:
                item = self._latest
                self._latest = None
            if item is None:
                time.sleep(0.002)
                continue

            frame_bgr, ts = item
            try:
                depth = self.vda.infer(frame_bgr)  # float32 HxW
                self.depth_ready.emit(depth, ts)
            except Exception as e:
                self.status_msg.emit(f"VDA error: {e}")

        self.status_msg.emit("VDA: worker stop")

    @QtCore.pyqtSlot()
    def stop(self):
        self._stop = True



class CaptureWorker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(np.ndarray, float, int, str)
    status_msg = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False
        self.device_index = 0
        self.width = 1920
        self.height = 1080
        self.fps = 60.0
        self.burnin = True
        self.tc = Timecode(self.fps)
        self.writer_worker: VideoWriterWorker = None
        self.recording = False
        self.frame_counter = 0

    def configure(self, device_index: int, width: int, height: int, fps: float, burnin: bool):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.burnin = burnin
        self.tc.set_fps(fps)

    def attach_writer(self, ww: VideoWriterWorker):
        self.writer_worker = ww

    @QtCore.pyqtSlot()
    def start(self):
        if self.running:
            return
        self.status_msg.emit(f"Open camera {self.device_index}…")
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            self.cap.release(); self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release(); self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_ANY)
        if not self.cap or not self.cap.isOpened():
            self.status_msg.emit("ERROR: Impossible to open camera.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self.cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        for _ in range(WARMUP_GRABS):
            self.cap.grab()
        ok, _ = self.cap.read()
        if not ok:
            self.status_msg.emit("ERROR: camera open but no image.")
            self.cap.release(); self.cap = None; return
        self.running = True
        self.frame_counter = 0
        self.status_msg.emit("Camera streaming.")
        QtCore.QTimer.singleShot(0, self._grab_loop)

    def _grab_loop(self):
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        ts_posix = time.time()
        if ok and frame is not None:
            self.frame_counter += 1
            tc_str = self.tc.from_timestamp(ts_posix)
            mono_ns = time.monotonic_ns()
            if self.burnin:
                vis = frame.copy()
                text1 = f"TC {tc_str}"
                text2 = datetime.fromtimestamp(ts_posix, tz=timezone.utc).strftime("UTC %Y-%m-%d %H:%M:%S.%f")[:-3]
                cv2.rectangle(vis, (5, 5), (520, 60), (0, 0, 0), -1)
                cv2.putText(vis, text1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(vis, text2, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                vis = frame
            self.frame_ready.emit(vis, ts_posix, mono_ns, tc_str)
            if self.recording and self.writer_worker is not None and self.writer_worker.opened:
                meta = {
                    "frame_idx": self.frame_counter,
                    "dropped": 0,
                    "tc": tc_str,
                    "utc": now_utc_iso(),
                    "local": now_local_iso(),
                    "mono_ns": mono_ns,
                    "w": frame.shape[1],
                    "h": frame.shape[0],
                }
                self.writer_worker.push((frame, meta))
        else:
            self.status_msg.emit("WARNING: frame lost.")
        QtCore.QTimer.singleShot(0, self._grab_loop)

    @QtCore.pyqtSlot()
    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release(); self.cap = None
        self.status_msg.emit("Camera stopped.")

# ============================ EXR Worker ============================ #

class ExrSequencePage(QtWidgets.QWidget):
    status_msg = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, vda_wrapper=None):
        super().__init__(parent)

        self.exr_dir = None
        self.exr_files = []
        self.meta_rows_seq = []          # list of rows (CSV order)
        self.meta_rows_by_frame = {}     # dict[frame] -> row if col frame is found
        self.meta_clip_name = None
        self.meta_path = None
        self._exr_tc_cache = {}

        self.vda = vda_wrapper   
        self.vda_enabled = False
        self.depth_cache = {}    # key = exr_path (or idx) -> depth float32
        self._last_req_key = None

        self.idx = 0
        self.playing = False

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)

        # ---- UI ----
        layout = QtWidgets.QHBoxLayout(self)

        # --- VDA thread (EXR) ---
        self.vda_thread = None
        self.vda_worker = None
        if self.vda is not None:
            self.vda_worker = VDAWorker(self.vda)
            self.vda_thread = QtCore.QThread(self)
            self.vda_worker.moveToThread(self.vda_thread)
            self.vda_thread.started.connect(self.vda_worker.run_loop)
            self.vda_worker.depth_ready.connect(self._on_vda_depth_ready)
            self.vda_worker.status_msg.connect(self._status)
            self.vda_thread.start()

        # Preview
        left = QtWidgets.QVBoxLayout()
        self.preview = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(960, 540)
        self.preview.setStyleSheet("background:#111; color:#aaa; border:1px solid #333;")
        left.addWidget(self.preview, 1)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.on_slider)
        left.addWidget(self.slider)

        self.info_line = QtWidgets.QLabel("No EXR sequence loaded.")
        self.info_line.setStyleSheet("color:#9cf")
        left.addWidget(self.info_line)

        layout.addLayout(left, 3)

        # Controls
        right = QtWidgets.QVBoxLayout()
        box = QtWidgets.QGroupBox("EXR Dataset")
        g = QtWidgets.QGridLayout(box)
        r = 0

        self.btn_open_exr = QtWidgets.QPushButton("Open folder EXR…")
        self.btn_open_csv = QtWidgets.QPushButton("Load CSV Meta…")
        self.btn_open_opti = QtWidgets.QPushButton("Load OptiTrack CSV…")
        self.btn_open_opti.setEnabled(False)
        self.btn_open_csv.setEnabled(False)
        g.addWidget(self.btn_open_opti, r, 0, 1, 2); r += 1

        g.addWidget(self.btn_open_exr, r, 0, 1, 2); r += 1
        g.addWidget(self.btn_open_csv, r, 0, 1, 2); r += 1

        g.addWidget(QtWidgets.QLabel("Read FPS"), r, 0)
        self.fps_box = QtWidgets.QDoubleSpinBox()
        self.fps_box.setRange(1.0, 240.0)
        self.fps_box.setDecimals(3)
        self.fps_box.setValue(23.976)
        g.addWidget(self.fps_box, r, 1); r += 1

        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.btn_stop = QtWidgets.QPushButton("■ Stop")
        self.btn_prev = QtWidgets.QPushButton("◀ Frame")
        self.btn_next = QtWidgets.QPushButton("Frame ▶")

        self.chk_vda = QtWidgets.QCheckBox("View VDA (Depth)")
        self.chk_vda.setChecked(False)

        self.chk_vda_cache = QtWidgets.QCheckBox("Cache depth")
        self.chk_vda_cache.setChecked(True)

        g.addWidget(self.chk_vda, r, 0, 1, 2); r += 1
        g.addWidget(self.chk_vda_cache, r, 0, 1, 2); r += 1

        g.addWidget(QtWidgets.QLabel("VDA downscale"), r, 0)
        self.vda_downscale = QtWidgets.QDoubleSpinBox()
        self.vda_downscale.setRange(0.10, 1.00)
        self.vda_downscale.setSingleStep(0.10)
        self.vda_downscale.setValue(0.50)
        g.addWidget(self.vda_downscale, r, 1); r += 1

        self.chk_vda.toggled.connect(self._on_toggle_vda)


        g.addWidget(self.btn_play, r, 0); g.addWidget(self.btn_stop, r, 1); r += 1
        g.addWidget(self.btn_prev, r, 0); g.addWidget(self.btn_next, r, 1); r += 1

        # Meta display
        self.meta_view = QtWidgets.QPlainTextEdit()
        self.meta_view.setReadOnly(True)
        self.meta_view.setMinimumHeight(220)

        right.addWidget(box)
        right.addWidget(QtWidgets.QLabel("Metadatas (current frame)"))
        right.addWidget(self.meta_view)

        # ---- Viewer 3D OptiTrack ----
        self.viewer3d = OptiTrack3DViewer(max_points=8000, mm_to_m=True)
        self.viewer3d.setMinimumHeight(260)
        right.addWidget(self.viewer3d, 1)  # stretch=1 

        layout.addLayout(right, 2)

        # Connections
        self.btn_open_exr.clicked.connect(self.open_exr_dir)
        self.btn_open_csv.clicked.connect(self.open_meta_csv)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop.clicked.connect(self.stop_play)
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_next.clicked.connect(lambda: self.step(+1))
        self.btn_open_opti.clicked.connect(self.open_optitrack_csv)


    # ---------- Loaders ----------
    def open_optitrack_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Charger OptiTrack CSV",
            str(self.exr_dir or Path.cwd()),
            "CSV (*.csv)"
        )
        if not path:
            return

        self.opti_path = Path(path)
        by_base, fps_est, sfpf, tc0, tc1 = self._parse_optitrack_csv(self.opti_path)

        self.opti_by_base_tc = by_base
        self.opti_fps = fps_est
        self.opti_subframes_per_frame = sfpf

        self._build_opti_alignment_cache()

        nkeys = len(self.opti_by_base_tc)
        msg = f"OptiTrack loaded: {self.opti_path.name} | baseTC={nkeys}"
        if self.opti_fps:
            msg += f" | fps≈{self.opti_fps:.3f}"
        if self.opti_subframes_per_frame:
            msg += f" | subframes/frame={self.opti_subframes_per_frame}"
        if tc0 and tc1:
            msg += f" | TC {tc0} → {tc1}"

        self._status(msg)
        self._render_current()

        if hasattr(self, "viewer3d") and self.viewer3d is not None:
            self.viewer3d.clear()



    def _pick_first(self, row: dict, keys):
        """
        Return 1re value not empty '--' from many keys possible.
        """
        if not row:
            return None

        # normalized -> original list of key
        norm_map = {}
        for kk in row.keys():
            if kk is None:
                continue
            k_clean = str(kk).replace("\ufeff", "").strip()
            nk = k_clean.lower()
            norm_map.setdefault(nk, []).append(kk)

        def _valid(v):
            v = "" if v is None else str(v).strip()
            return v and v not in ("--", "-", "nan", "None")

        for wanted in keys:
            w = str(wanted).replace("\ufeff", "").strip().lower()

            # 1) match exact normalized
            if w in norm_map:
                for orig in norm_map[w]:
                    v = row.get(orig, "")
                    if _valid(v):
                        return str(v).strip()

            # 2) fallback: content (ex: "Master TC Time Base", "Master TC__2", etc.)
            for nk, orig_list in norm_map.items():
                if w in nk:
                    for orig in orig_list:
                        v = row.get(orig, "")
                        if _valid(v):
                            return str(v).strip()

        return None




    def _smpte_to_frames(self, tc: str, fps_nominal: int) -> int:
        """HH:MM:SS:FF -> total frames (approx, non-drop)."""
        try:
            hh, mm, ss, ff = tc.strip().split(":")
            hh = int(hh); mm = int(mm); ss = int(ss); ff = int(ff)
            return (((hh * 60 + mm) * 60 + ss) * fps_nominal) + ff
        except Exception:
            return 0


    def _frames_to_smpte(self, total_frames: int, fps_nominal: int) -> str:
        """total frames -> HH:MM:SS:FF (approx, non-drop)."""
        if fps_nominal <= 0:
            fps_nominal = 24
        total_frames = max(0, int(total_frames))
        ff = total_frames % fps_nominal
        total_seconds = total_frames // fps_nominal
        ss = total_seconds % 60
        total_minutes = total_seconds // 60
        mm = total_minutes % 60
        hh = (total_minutes // 60) % 24
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


    def _infer_nominal_fps(self) -> int:
        """
        For TC SMPTE, take a fps nominal (24/25/30/60/120...).
        23.976 -> 24 ; 29.97 -> 30 ; etc.
        """
        fps = float(self.fps_box.value())
        candidates = [24, 25, 30, 48, 50, 60, 96, 100, 120, 240]
        return min(candidates, key=lambda x: abs(x - fps))


    def _draw_panel_bottom_left(self, img_bgr8: np.ndarray, lines, pad=10):
        """ Draw pannel."""
        if img_bgr8 is None or not lines:
            return img_bgr8

        h, w = img_bgr8.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        line_h = 18

        max_w = 0
        for t in lines:
            (tw, _), _ = cv2.getTextSize(t, font, font_scale, thickness)
            max_w = max(max_w, tw)

        box_w = max_w + 2 * pad
        box_h = len(lines) * line_h + 2 * pad

        x1, y2 = 10, h - 10
        x2, y1 = x1 + box_w, y2 - box_h

        x2 = min(x2, w - 10)
        y1 = max(y1, 10)

        # black pannel
        cv2.rectangle(img_bgr8, (x1, y1), (x2, y2), (10, 10, 10), -1)
        cv2.rectangle(img_bgr8, (x1, y1), (x2, y2), (60, 60, 60), 1)

        y = y1 + pad + 14
        for t in lines:
            cv2.putText(img_bgr8, t, (x1 + pad, y), font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
            y += line_h

        return img_bgr8



    def open_exr_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose a sequence EXR folder", str(Path.cwd()))
        if not d:
            return
        self.exr_dir = Path(d)
        files = sorted(glob.glob(str(self.exr_dir / "*.exr")), key=natural_key)
        if not files:
            QtWidgets.QMessageBox.warning(self, "EXR", "No file .exr found in folder.")
            return

        self.exr_files = files
        self.exr_frame_nums = [frame_num_from_exr_filename(p) for p in self.exr_files]
        self.idx = 0
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self.exr_files) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        self.btn_open_csv.setEnabled(True)
        self.btn_open_opti.setEnabled(True)

        self.meta_rows_seq = []
        self.meta_rows_by_frame = {}
        self.meta_clip_name = None
        self.meta_path = None

        self.opti_path = None
        self.opti_by_base_tc = {}   # dict["HH:MM:SS:FF"] -> list of samples (subframes)
        self.opti_fps = None
        self.opti_subframes_per_frame = None


        self._render_current()
        self._build_opti_alignment_cache()

        self._status(f"EXR: {len(self.exr_files)} loaded frames since {self.exr_dir.name}")

        if hasattr(self, "viewer3d") and self.viewer3d is not None:
            self.viewer3d.clear()



    def open_meta_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load CSV Meta Extract",
            str(self.exr_dir or Path.cwd()),
            "CSV (*.csv)"
        )
        if not path:
            return

        self.meta_path = Path(path)

        # Take outputs
        self.meta_rows_seq, self.meta_rows_by_frame, self.meta_clip_name = self._parse_meta_csv(self.meta_path)

        self._build_opti_alignment_cache()

        # Validate clip CSV vs EXR
        exr_prefix = Path(self.exr_files[0]).name.split(".")[0] if self.exr_files else ""
        clip_name = self.meta_clip_name

        if clip_name and exr_prefix and (exr_prefix not in clip_name) and (clip_name not in exr_prefix):
            QtWidgets.QMessageBox.warning(
                self,
                "Warning",
                f"The CSV file seems to be for another clip.\n\nEXR: {exr_prefix}\nCSV: {clip_name}\n\n"
                "If you continue, the alignement frame→meta can be wrong."
            )

        self._render_current()

        nseq = len(self.meta_rows_seq)
        nmap = len(self.meta_rows_by_frame)
        self._status(f"CSV loaded: {self.meta_path.name}  (lines={nseq}, index frame={nmap})")


    def _parse_meta_csv(self, csv_path: Path):
        """
        Return:
        meta_rows_seq: list[dict]          (from line order)
        meta_rows_by_frame: dict[int,dict] (if col frame is found)
        clip_name: str|None
        """
        meta_rows_seq = []
        meta_rows_by_frame = {}
        clip_name = None

        def detect_delimiter(sample: str) -> str:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
                return dialect.delimiter
            except Exception:
                counts = {d: sample.count(d) for d in [",", "\t", ";", "|"]}
                delim = max(counts, key=counts.get)
                return delim if counts[delim] > 0 else ","

        def uniquify_headers(raw_headers):
            from collections import Counter
            seen = Counter()
            out = []
            for h in raw_headers:
                h2 = str(h).replace("\ufeff", "").strip()
                seen[h2] += 1
                if seen[h2] == 1:
                    out.append(h2)
                else:
                    out.append(f"{h2}__{seen[h2]}")
            return out

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            sample = f.read(8192)
            f.seek(0)
            delim = detect_delimiter(sample)

            reader = csv.reader(f, delimiter=delim)
            raw_headers = next(reader, [])
            headers = uniquify_headers(raw_headers)

            # col frame available
            frame_col = None
            priority = ["Frame", "frame", "Index", "index", "Frame Number", "Frame Num", "Frame#", "frame_idx"]
            for p in priority:
                for h in headers:
                    if h.lower() == p.lower():
                        frame_col = h
                        break
                if frame_col:
                    break
            if frame_col is None:
                for h in headers:
                    if re.search(r"\b(frame|index)\b", h, re.IGNORECASE):
                        frame_col = h
                        break

            for row_vals in reader:
                if not row_vals:
                    continue

                # pad if row shorter
                if len(row_vals) < len(headers):
                    row_vals = row_vals + [""] * (len(headers) - len(row_vals))

                clean_row = {}
                for i in range(len(headers)):
                    v = row_vals[i] if i < len(row_vals) else ""
                    clean_row[headers[i]] = "" if v is None else str(v).strip()

                meta_rows_seq.append(clean_row)

                # Clip name
                if clip_name is None:
                    for k in ["Camera Clip Name", "Clip Name", "Clip", "Source Clip Name", "Name"]:
                        v = self._pick_first(clean_row, [k])
                        if v:
                            clip_name = v
                            break

                # Index per frame
                if frame_col is not None:
                    vfi = clean_row.get(frame_col, "")
                    if vfi != "":
                        try:
                            fi = int(float(vfi))
                            meta_rows_by_frame[fi] = clean_row
                        except Exception:
                            pass

        return meta_rows_seq, meta_rows_by_frame, clip_name

    def _parse_optitrack_csv(self, csv_path: Path):
        """
        Parse CSV OptiTrack (Motive Take export).
        """
        import statistics

        by_base = {}
        times = []
        tc_start = None
        tc_end = None

        with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            # Forward to line header
            header = None
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("Frame,Time"):
                    header = line
                    break
            if header is None:
                return {}, None, None, None, None

            reader = csv.reader(f, delimiter=",")
            for row in reader:
                if not row or len(row) < 10:
                    continue

                # fields
                try:
                    tsec = float(row[1])
                except Exception:
                    continue

                tc_full = str(row[2]).strip()
                if not tc_full:
                    continue

                # split subframe
                if "." in tc_full:
                    base_tc, sf_str = tc_full.split(".", 1)
                    try:
                        subf = int(sf_str)
                    except Exception:
                        subf = 0
                else:
                    base_tc = tc_full
                    subf = 0

                base_tc = base_tc.strip()

                try:
                    qx = float(row[3]); qy = float(row[4]); qz = float(row[5]); qw = float(row[6])
                    px = float(row[7]); py = float(row[8]); pz = float(row[9])
                except Exception:
                    continue

                d = {
                    "time_s": tsec,
                    "tc_full": tc_full,
                    "base_tc": base_tc,
                    "subframe": subf,
                    "quat": (qx, qy, qz, qw),
                    "pos_mm": (px, py, pz),
                }
                by_base.setdefault(base_tc, []).append(d)

                times.append(tsec)
                if tc_start is None:
                    tc_start = tc_full
                tc_end = tc_full

        # Estimate FPS via Time(Seconds)
        fps_est = None
        if len(times) >= 3:
            dts = [times[i+1] - times[i] for i in range(len(times)-1) if (times[i+1] - times[i]) > 0]
            if dts:
                med = statistics.median(dts)
                if med > 0:
                    fps_est = 1.0 / med

        # subframes_per_frame = max count observed per base_tc
        subframes_per_frame = None
        if by_base:
            subframes_per_frame = max(len(v) for v in by_base.values())

        # Sort each list per subframe
        for k in list(by_base.keys()):
            by_base[k].sort(key=lambda x: x["subframe"])

        return by_base, fps_est, subframes_per_frame, tc_start, tc_end
    
    def _get_tc_for_exr_index(self, i: int):
        """Return a TC reference for index EXR i (EXR TC if available if not CSV meta)."""
        if not self.exr_files or i < 0 or i >= len(self.exr_files):
            return None

        exr_path = self.exr_files[i]
        tc_exr = exr_read_timecode(exr_path)
        if tc_exr:
            return tc_exr

        # fallback meta CSV (if available)
        frame_num = self.exr_frame_nums[i] if hasattr(self, "exr_frame_nums") else (i + 1)
        row = None
        if self.meta_rows_by_frame:
            row = self.meta_rows_by_frame.get(frame_num, None) or self.meta_rows_by_frame.get(frame_num - 1, None)
        if row is None and self.meta_rows_seq:
            if 1 <= frame_num <= len(self.meta_rows_seq):
                row = self.meta_rows_seq[frame_num - 1]
            elif 0 <= i < len(self.meta_rows_seq):
                row = self.meta_rows_seq[i]

        if row is not None:
            tc_meta = self._pick_first(row, ["Master TC", "Source TC", "Timecode", "Record TC", "TC"])
            return tc_meta
        return None


    def _build_opti_alignment_cache(self):
        """
        Pre-compute for each frame EXR i:
          - position mm (mid subframe) or None
          - quaternion (mid subframe) or None
        """
        self.opti_pose_by_exr_index = [None] * (len(self.exr_files) if self.exr_files else 0)

        if not self.exr_files or not getattr(self, "opti_by_base_tc", None):
            return

        for i in range(len(self.exr_files)):
            tc_ref = self._get_tc_for_exr_index(i)
            if not tc_ref:
                continue
            base_tc = str(tc_ref).split(".")[0].strip()
            samples = self.opti_by_base_tc.get(base_tc, None)
            if not samples:
                continue
            mid = samples[len(samples)//2]
            px, py, pz = mid["pos_mm"]
            qx, qy, qz, qw = mid["quat"]
            self.opti_pose_by_exr_index[i] = {
                "pos_mm": (px, py, pz),
                "quat": (qx, qy, qz, qw),
                "base_tc": base_tc,
                "subframe": int(mid.get("subframe", 0)),
                "n_samples": int(len(samples)),
            }

    # ---------- VDA Cache ----------
    @QtCore.pyqtSlot(np.ndarray, float)
    def _on_vda_depth_ready(self, depth: np.ndarray, ts: float):
        # Depth received from last request
        key = self._last_req_key
        if not key:
            return

        if self.chk_vda_cache.isChecked():
            self.depth_cache[key] = depth
        else:
            # no cache: temporally take
            self.depth_cache = {key: depth}

        # if still on frame, redraw
        cur_key = str(self.exr_files[self.idx]) if self.exr_files else None
        if cur_key == key and self.vda_enabled:
            self._render_current()


    # ---------- Playback ----------
    def toggle_play(self):
        if not self.exr_files:
            return
        self.playing = not self.playing
        if self.playing:
            self.btn_play.setText("⏸ Pause")
            self._arm_timer()
        else:
            self.btn_play.setText("▶ Play")
            self.timer.stop()

    def stop_play(self):
        self.playing = False
        self.btn_play.setText("▶ Play")
        self.timer.stop()

    def _arm_timer(self):
        fps = float(self.fps_box.value())
        interval_ms = max(1, int(1000.0 / fps))
        self.timer.start(interval_ms)

    def _tick(self):
        if not self.exr_files:
            return
        self.idx += 1
        if self.idx >= len(self.exr_files):
            self.idx = len(self.exr_files) - 1
            self.stop_play()
            return
        self.slider.blockSignals(True)
        self.slider.setValue(self.idx)
        self.slider.blockSignals(False)
        self._render_current()

    def step(self, delta: int):
        if not self.exr_files:
            return
        self.idx = int(np.clip(self.idx + delta, 0, len(self.exr_files) - 1))
        self.slider.blockSignals(True)
        self.slider.setValue(self.idx)
        self.slider.blockSignals(False)
        self._render_current()

    def on_slider(self, v: int):
        self.idx = int(v)
        self._render_current()

    def _on_toggle_vda(self, on: bool):
        self.vda_enabled = bool(on) and (self.vda_worker is not None)
        self._render_current()


    # ---------- Render ----------
    def _render_current(self):
        if not self.exr_files:
            self.preview.setText("No EXR sequence loaded.")
            return

        exr_path = self.exr_files[self.idx]
        tc_exr = self._exr_tc_cache.get(exr_path)
        if tc_exr is None:
            tc_exr = exr_read_timecode(exr_path)
            self._exr_tc_cache[exr_path] = tc_exr

        # frame_num from EXR (00000001 -> 1) before usage
        frame_num = self.exr_frame_nums[self.idx] if hasattr(self, "exr_frame_nums") else (self.idx + 1)

        exr = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
        if exr is None:
            self.preview.setText(
                f"Error reading EXR:\n{Path(exr_path).name}\n\n"
                f"(Validate OPENCV_IO_ENABLE_OPENEXR=1)"
            )
            self.info_line.setText(f"[{self.idx+1}/{len(self.exr_files)}] frame_num={frame_num}  |  {Path(exr_path).name}")
            return

        # EXR -> exr3
        if exr.ndim == 3 and exr.shape[2] >= 3:
            exr3 = exr[:, :, :3]
        else:
            exr3 = exr

        # --- build key cache (per path) ---
        cache_key = str(exr_path)

        # --- VDA path: request depth async ---
        if self.vda_enabled and self.vda_worker is not None:
            depth = self.depth_cache.get(cache_key, None)

            # if not in cach: submit job to worker
            if depth is None:
                # image for VDA (downscale VDA)
                bgr8_for_vda = exr_float_to_vda_bgr8(exr3)
                ds = float(self.vda_downscale.value())
                if ds < 0.999:
                    h1, w1 = bgr8_for_vda.shape[:2]
                    bgr8_for_vda = cv2.resize(
                        bgr8_for_vda,
                        (max(16, int(w1*ds)), max(16, int(h1*ds))),
                        interpolation=cv2.INTER_AREA
                    )

                # memorize current request
                self._last_req_key = cache_key

                # submit async (ts = time.time())
                QtCore.QMetaObject.invokeMethod(
                    self.vda_worker,
                    "submit",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(np.ndarray, bgr8_for_vda),
                    QtCore.Q_ARG(float, float(time.time()))
                )

                # fallback: show RGB preview during compute
                vis8 = exr_to_preview_bgr8(exr3)
                vis8 = self._draw_panel_bottom_left(vis8, panel_lines + ["VDA: computing..."])
            else:
                # depth ready -> show depth
                vis8 = depth_to_vis_u8(depth)
                vis8 = self._draw_panel_bottom_left(vis8, panel_lines + ["VDA: ON"])
        else:
            # normal RGB
            vis8 = exr_to_preview_bgr8(exr3)
            vis8 = self._draw_panel_bottom_left(vis8, panel_lines)


        # >>> downscale for preview (ex: width max 1600)
        max_w = 1600
        h0, w0 = exr3.shape[:2]
        if w0 > max_w:
            scale = max_w / float(w0)
            exr3 = cv2.resize(exr3, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)

        vis8 = exr_to_preview_bgr8(exr3)
        if vis8 is None:
            self.preview.setText("Error: conversion preview EXR.")
            return

        # -------------------- search row meta of the frame --------------------
        frame_num = self.exr_frame_nums[self.idx] if hasattr(self, "exr_frame_nums") else (self.idx + 1)

        row = None
        if self.meta_rows_by_frame:
            row = self.meta_rows_by_frame.get(frame_num, None)
            if row is None:
                row = self.meta_rows_by_frame.get(frame_num - 1, None)  # CSV can be 0-index
        if row is None and self.meta_rows_seq:
            if 1 <= frame_num <= len(self.meta_rows_seq):
                row = self.meta_rows_seq[frame_num - 1]
            elif 0 <= self.idx < len(self.meta_rows_seq):
                row = self.meta_rows_seq[self.idx]

        # -------------------- construct lines for left-down pannel --------------------
        exr_prefix = Path(exr_path).name.split(".")[0]
        clip_name = None
        if row is not None:
            clip_name = self._pick_first(row, ["Camera Clip Name", "Clip Name", "Source Clip Name", "Name"])
        if not clip_name:
            clip_name = self.meta_clip_name or exr_prefix

        fps_nom = self._infer_nominal_fps()

        # --- TC "media" = timecode from position (frame_num) + fps nominal ---
        # 00:00:00:00 at frame 1 (so frame_num-1 in frames)
        tc_media = self._frames_to_smpte(max(0, frame_num - 1), fps_nom)

        # lens infos
        lens = self._pick_first(row, ["Lens Model"]) if row else None
        focal = self._pick_first(row, ["Lens Focal Length"]) if row else None
        focus = self._pick_first(row, ["Lens Focus Distance"]) if row else None
        iris  = self._pick_first(row, ["Lens Iris"]) if row else None
        expo  = self._pick_first(row, ["Exposure Time"]) if row else None
        shut  = self._pick_first(row, ["Shutter Angle"]) if row else None
        iso   = self._pick_first(row, ["Exposure Index ASA"]) if row else None

        panel_lines = []
        panel_lines.append(f"{clip_name}")
        panel_lines.append(f"Frame {frame_num:06d} / {len(self.exr_files)}")

        # TC metadata if available (CSV)
        tc_meta = None
        if row is not None:
            tc_meta = self._pick_first(row, ["Master TC", "Source TC", "Timecode", "Record TC", "TC"])

        # --- LEft: TC from EXR only ---
        if tc_exr:
            panel_lines.append(f"TC {tc_exr}")
        else:
            panel_lines.append("TC (EXR absent)")
            # if need visual fallback, uncomment:
            # panel_lines.append(f"TC (media) {tc_media}")

        # Lens / expo
        if lens:
            panel_lines.append(f"Lens {lens}")

        bits1 = []
        if focal: bits1.append(f"Focal {focal}")
        if focus: bits1.append(f"Focus {focus}")
        if iris:  bits1.append(f"Iris {iris}")
        if bits1:
            panel_lines.append(" | ".join(bits1))

        bits2 = []
        if expo: bits2.append(f"Exp {expo}")
        if shut: bits2.append(f"Sh {shut}")
        if iso:  bits2.append(f"ISO {iso}")
        if bits2:
            panel_lines.append(" | ".join(bits2))


        # Draw black pannel left/down
        vis8 = self._draw_panel_bottom_left(vis8, panel_lines)

        # -------------------- Show image --------------------
        h, w = vis8.shape[:2]
        qimg = QtGui.QImage(vis8.data, w, h, vis8.strides[0], QtGui.QImage.Format.Format_BGR888)


        self.preview.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))

        # Info + meta
        name = Path(exr_path).name

        # frame_num from name EXR (00000001 -> 1)
        frame_num = self.exr_frame_nums[self.idx] if hasattr(self, "exr_frame_nums") else (self.idx + 1)

        self.info_line.setText(f"[{self.idx+1}/{len(self.exr_files)}] frame_num={frame_num}  |  {name}")

        # --- Metadatas: Show only lens/expo importants ---
        IMPORTANT_FIELDS = [
            # Lens (prior / real time calib)
            "Lens Model",
            "Lens Serial Number",
            "Lens Focal Length",
            "Lens Focus Distance",
            "Lens Iris",
            "Lens Squeeze",

            # Exposition (prior/validation)
            "Exposure Time",
            "Shutter Angle",
            "Exposure Index ASA",
            "White Balance",
            "Sensor FPS",
            "Project FPS",

            # Timecode
            "Master TC",
            "Source TC",
            "Timecode",
        ]

        row = None
        if self.meta_rows_by_frame:
            row = self.meta_rows_by_frame.get(frame_num, None)
            if row is None:
                row = self.meta_rows_by_frame.get(frame_num - 1, None)  # CSV can be 0-index
        if row is None and self.meta_rows_seq:
            if 1 <= frame_num <= len(self.meta_rows_seq):
                row = self.meta_rows_seq[frame_num - 1]
            elif 0 <= self.idx < len(self.meta_rows_seq):
                row = self.meta_rows_seq[self.idx]

        lines = []

        # 1) TC EXR
        lines.append(f"TC (EXR): {tc_exr if tc_exr else '(absent)'}")

        # 2) TC metadata (CSV)
        tc_meta = None
        if row is not None:
            tc_meta = self._pick_first(row, ["Master TC", "Source TC", "Timecode", "Record TC", "TC"])
        lines.append(f"TC (metadata): {tc_meta if tc_meta else '(absent from CSV for this frame)'}")

        # 3) Difference ΔTC (meta - exr)
        if tc_exr and tc_meta:
            fps_nom = self._infer_nominal_fps()
            exr_frames  = self._smpte_to_frames(tc_exr, fps_nom)
            meta_frames = self._smpte_to_frames(tc_meta, fps_nom)
            delta = meta_frames - exr_frames  

            # Show in frames + SMPTE
            sign = "+" if delta >= 0 else "-"
            delta_abs = abs(delta)
            delta_tc = self._frames_to_smpte(delta_abs, fps_nom)

            lines.append(f"ΔTC (meta - exr): {sign}{delta} frames  ({sign}{delta_tc})")
        else:
            lines.append("ΔTC (meta - exr): (impossible — TC EXR or metadata missing)")

        # 4) OptiTrack (if loaded)
        tc_ref = tc_exr or tc_meta  # Take EXR if not metadata
        if tc_ref and self.opti_by_base_tc:
            base_tc = str(tc_ref).split(".")[0].strip()  # EXR is without .sf, but safe
            samples = self.opti_by_base_tc.get(base_tc, None)

            if samples:
                # choose subframe from middle
                mid = samples[len(samples)//2]
                sfpf = int(self.opti_subframes_per_frame or max(1, len(samples)))

                opti_tc_full = f"{base_tc}.{int(mid['subframe']):02d}"
                lines.append(f"TC (OptiTrack): {opti_tc_full}  ({len(samples)} échant./frame)")

                # ΔTC opti - exr : exr is at subframe 0 per convention
                delta_sf = int(mid["subframe"]) - 0
                delta_frames = float(delta_sf) / float(sfpf) if sfpf > 0 else 0.0
                sign = "+" if delta_sf >= 0 else "-"
                lines.append(f"ΔTC (opti - exr): {sign}{abs(delta_sf)} subframes  ({sign}{abs(delta_frames):.3f} frames)")

                # Pose OptiTrack
                px, py, pz = mid["pos_mm"]
                qx, qy, qz, qw = mid["quat"]
                lines.append(f"Opti pos (mm): [{px:.3f}, {py:.3f}, {pz:.3f}]")
                lines.append(f"Opti quat (x,y,z,w): [{qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}]")

                # --- push to 3D viewer ---
                if hasattr(self, "viewer3d") and self.viewer3d is not None:
                    # 1) trajectory = all poses till idx
                    if hasattr(self, "opti_pose_by_exr_index") and self.opti_pose_by_exr_index:
                        pts = []
                        for k in range(0, self.idx + 1):
                            d = self.opti_pose_by_exr_index[k]
                            pts.append(d["pos_mm"] if d else None)
                        self.viewer3d.set_trajectory_mm(pts, reset_origin=True)

                    # 2) current pose = point + frustum (if toggle)
                    self.viewer3d.set_current_pose_mm(px, py, pz, qx, qy, qz, qw)



            else:
                lines.append(f"TC (OptiTrack): (no sample for {base_tc})")
        elif self.opti_by_base_tc and not tc_ref:
            lines.append("TC (OptiTrack): (no TC EXR/metadata for matching)")

        if row is not None:
            for k in IMPORTANT_FIELDS:
                if k in row:
                    v = str(row.get(k, "")).strip()
                    if v and v not in ("--", "-", "nan", "None"):
                        # éviter de dupliquer les TC déjà affichés
                        if k in ("Master TC", "Source TC", "Timecode"):
                            continue
                        lines.append(f"{k}: {v}")

        meta_txt = "\n".join(lines) if lines else "(No field lens/expo found for this frame — CSV ok?)"
        self.meta_view.setPlainText(meta_txt)



    def _status(self, msg: str):
        self.info_line.setText(msg)



# ============================ Metadata Panel ============================ #
class MetadataPanel(QtWidgets.QGroupBox):
    def __init__(self):
        super().__init__("Metadatas session / dataset")
        grid = QtWidgets.QGridLayout(self)
        r = 0
        # IDs
        self.video_id = QtWidgets.QSpinBox(); self.video_id.setRange(0, 1_000_000); self.video_id.setValue(1)
        self.scene_id = QtWidgets.QSpinBox(); self.scene_id.setRange(0, 1_000_000); self.scene_id.setValue(1)
        self.take_id = QtWidgets.QSpinBox(); self.take_id.setRange(0, 1_000_000); self.take_id.setValue(1)
        grid.addWidget(QtWidgets.QLabel("video_id"), r, 0); grid.addWidget(self.video_id, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("scene_id"), r, 0); grid.addWidget(self.scene_id, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("take_id"), r, 0); grid.addWidget(self.take_id, r, 1); r += 1
        # Movement / speed
        self.mouvement_cam = QtWidgets.QLineEdit("fix")
        self.vitesse_cam = QtWidgets.QComboBox(); self.vitesse_cam.addItems(["", "slow", "normal", "fast"]) ; self.vitesse_cam.setCurrentText("normale")
        grid.addWidget(QtWidgets.QLabel("mouvement_cam"), r, 0); grid.addWidget(self.mouvement_cam, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("vitesse_cam"), r, 0); grid.addWidget(self.vitesse_cam, r, 1); r += 1
        # Goalds / distances / aperture
        self.objectif = QtWidgets.QLineEdit("")
        self.focal_mm = QtWidgets.QDoubleSpinBox(); self.focal_mm.setRange(0.1, 1000.0); self.focal_mm.setDecimals(2); self.focal_mm.setValue(35.20)
        self.focus_distance_m = QtWidgets.QDoubleSpinBox(); self.focus_distance_m.setRange(0.0, 100.0); self.focus_distance_m.setDecimals(3); self.focus_distance_m.setValue(2.800)
        self.aperture_f = QtWidgets.QDoubleSpinBox(); self.aperture_f.setRange(0.5, 32.0); self.aperture_f.setDecimals(2); self.aperture_f.setValue(2.80)
        grid.addWidget(QtWidgets.QLabel("objectif"), r, 0); grid.addWidget(self.objectif, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("focal_mm"), r, 0); grid.addWidget(self.focal_mm, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("focus_distance_m"), r, 0); grid.addWidget(self.focus_distance_m, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("aperture_f"), r, 0); grid.addWidget(self.aperture_f, r, 1); r += 1
        # Éclairage / LED / contenu
        self.eclairage = QtWidgets.QLineEdit("pas de contenu")
        self.led_brightness = QtWidgets.QSpinBox(); self.led_brightness.setRange(0, 100); self.led_brightness.setValue(60)
        self.contenu_LED = QtWidgets.QLineEdit("pas de contenu")
        grid.addWidget(QtWidgets.QLabel("eclairage"), r, 0); grid.addWidget(self.eclairage, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("led_brightness"), r, 0); grid.addWidget(self.led_brightness, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("contenu_LED"), r, 0); grid.addWidget(self.contenu_LED, r, 1); r += 1
        # Scène / conditions / qualité
        self.scene_type = QtWidgets.QLineEdit("")
        self.conditions = QtWidgets.QLineEdit("pas de contenu")
        self.qualite_take = QtWidgets.QComboBox(); self.qualite_take.addItems(["", "bonne", "moyenne", "mauvaise"]) ; self.qualite_take.setCurrentText("bonne")
        grid.addWidget(QtWidgets.QLabel("scène_type"), r, 0); grid.addWidget(self.scene_type, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("conditions (séparées par des virgules)"), r, 0); grid.addWidget(self.conditions, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("qualité_take"), r, 0); grid.addWidget(self.qualite_take, r, 1); r += 1
        # Chemin/notes
        self.chemin_cam = QtWidgets.QLineEdit("")
        self.notes = QtWidgets.QLineEdit("")
        grid.addWidget(QtWidgets.QLabel("chemin_cam"), r, 0); grid.addWidget(self.chemin_cam, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("notes"), r, 0); grid.addWidget(self.notes, r, 1); r += 1
        # Date / paths / GT
        self.date_capture = QtWidgets.QDateEdit(QtCore.QDate.currentDate()); self.date_capture.setCalendarPopup(True)
        self.path_video = QtWidgets.QLineEdit(""); self.path_frames = QtWidgets.QLineEdit("")
        self.has_gt_pose = QtWidgets.QCheckBox("has_gt_pose"); self.has_gt_pose.setChecked(True)
        grid.addWidget(QtWidgets.QLabel("date_capture"), r, 0); grid.addWidget(self.date_capture, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("path_video"), r, 0); grid.addWidget(self.path_video, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("path_frames"), r, 0); grid.addWidget(self.path_frames, r, 1); r += 1
        grid.addWidget(self.has_gt_pose, r, 0, 1, 2); r += 1
        # Camera / Lens / Exposition (sans tags_auto / camera_make / lens_make)
        self.camera_model = QtWidgets.QLineEdit("ELP")
        self.lens_model = QtWidgets.QLineEdit("")
        self.shutter_angle = QtWidgets.QDoubleSpinBox(); self.shutter_angle.setRange(0.0, 360.0); self.shutter_angle.setDecimals(1); self.shutter_angle.setValue(180.0)
        self.iso = QtWidgets.QSpinBox(); self.iso.setRange(0, 128000); self.iso.setValue(800)
        self.white_balance_K = QtWidgets.QSpinBox(); self.white_balance_K.setRange(1000, 20000); self.white_balance_K.setValue(5600)
        grid.addWidget(QtWidgets.QLabel("camera_model"), r, 0); grid.addWidget(self.camera_model, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("lens_model"), r, 0); grid.addWidget(self.lens_model, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("shutter_angle"), r, 0); grid.addWidget(self.shutter_angle, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("iso"), r, 0); grid.addWidget(self.iso, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("white_balance_K"), r, 0); grid.addWidget(self.white_balance_K, r, 1); r += 1

        # Auto path_frames quand scene/take changent
        self.scene_id.valueChanged.connect(self._update_paths)
        self.take_id.valueChanged.connect(self._update_paths)
        self._update_paths()

    def _update_paths(self):
        s = self.scene_id.value(); t = self.take_id.value()
        self.path_frames.setText(f"frames_1080p/scene_{s}/take_{t}/")

    def to_dict(self):
        def split_csv(text):
            return [x.strip() for x in text.split(',') if x.strip()]
        return {
            "video_id": int(self.video_id.value()),
            "scene_id": int(self.scene_id.value()),
            "take_id": int(self.take_id.value()),
            "mouvement_cam": self.mouvement_cam.text(),
            "vitesse_cam": self.vitesse_cam.currentText(),
            "objectif": self.objectif.text(),
            "focal_mm": float(self.focal_mm.value()),
            "focus_distance_m": float(self.focus_distance_m.value()),
            "aperture_f": float(self.aperture_f.value()),
            "eclairage": self.eclairage.text(),
            "led_brightness": int(self.led_brightness.value()),
            "contenu_LED": self.contenu_LED.text(),
            "scène_type": self.scene_type.text(),
            "conditions": split_csv(self.conditions.text()),
            "qualité_take": self.qualite_take.currentText(),
            "chemin_cam": self.chemin_cam.text(),
            "notes": self.notes.text(),
            "date_capture": self.date_capture.date().toString("yyyy-MM-dd"),
            "path_video": self.path_video.text(),
            "path_frames": self.path_frames.text(),
            "has_gt_pose": bool(self.has_gt_pose.isChecked()),
            "camera_model": self.camera_model.text(),
            "lens_model": self.lens_model.text(),
            "shutter_angle": float(self.shutter_angle.value()),
            "iso": int(self.iso.value()),
            "white_balance_K": int(self.white_balance_K.value()),
        }

    def set_path_video(self, p: str):
        self.path_video.setText(p)

    def load_from_dict(self, data: dict):
        # Charge partiellement si des clés manquent
        self.video_id.setValue(int(data.get("video_id", self.video_id.value())))
        self.scene_id.setValue(int(data.get("scene_id", self.scene_id.value())))
        self.take_id.setValue(int(data.get("take_id", self.take_id.value())))
        self.mouvement_cam.setText(str(data.get("mouvement_cam", self.mouvement_cam.text())))
        self.vitesse_cam.setCurrentText(str(data.get("vitesse_cam", self.vitesse_cam.currentText())))
        self.objectif.setText(str(data.get("objectif", self.objectif.text())))
        if "focal_mm" in data: self.focal_mm.setValue(float(data["focal_mm"]))
        if "focus_distance_m" in data: self.focus_distance_m.setValue(float(data["focus_distance_m"]))
        if "aperture_f" in data: self.aperture_f.setValue(float(data["aperture_f"]))
        self.eclairage.setText(str(data.get("eclairage", self.eclairage.text())))
        self.led_brightness.setValue(int(data.get("led_brightness", self.led_brightness.value())))
        self.contenu_LED.setText(str(data.get("contenu_LED", self.contenu_LED.text())))
        self.scene_type.setText(str(data.get("scène_type", self.scene_type.text())))
        self.conditions.setText(", ".join(data.get("conditions", [])) if isinstance(data.get("conditions", []), list) else str(data.get("conditions", "")))
        self.qualite_take.setCurrentText(str(data.get("qualité_take", self.qualite_take.currentText())))
        self.chemin_cam.setText(str(data.get("chemin_cam", self.chemin_cam.text())))
        self.notes.setText(str(data.get("notes", self.notes.text())))
        if "date_capture" in data:
            try:
                d = QtCore.QDate.fromString(str(data["date_capture"]), "yyyy-MM-dd")
                if d.isValid(): self.date_capture.setDate(d)
            except Exception:
                pass
        self.path_video.setText(str(data.get("path_video", self.path_video.text())))
        self.path_frames.setText(str(data.get("path_frames", self.path_frames.text())))
        self.has_gt_pose.setChecked(bool(data.get("has_gt_pose", self.has_gt_pose.isChecked())))
        self.camera_model.setText(str(data.get("camera_model", self.camera_model.text())))
        self.lens_model.setText(str(data.get("lens_model", self.lens_model.text())))
        if "shutter_angle" in data: self.shutter_angle.setValue(float(data["shutter_angle"]))
        if "iso" in data: self.iso.setValue(int(data["iso"]))
        if "white_balance_K" in data: self.white_balance_K.setValue(int(data["white_balance_K"]))


# ============================ Page Calibrate Stereo (INTÉGRÉE) ============================ #
class CalibrateStereoPage(QtWidgets.QWidget):
    status_msg = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)

        # Buffers de calibration
        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []
        self.gray_size = None
        self.last_capture_ms = 0
        self.calib_data = None  # dict YAML prêt à enregistrer

        # Layout principal
        layout = QtWidgets.QHBoxLayout(self)

        # ---- Prévisualisation ----
        left = QtWidgets.QVBoxLayout()
        self.preview = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(960, 540)
        self.preview.setStyleSheet("background:#111; color:#aaa; border:1px solid #333;")
        left.addWidget(self.preview, 1)
        self.progress_lbl = QtWidgets.QLabel("0 capture / 15"); self.progress_lbl.setStyleSheet("color:#9cf")
        left.addWidget(self.progress_lbl)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(140)
        left.addWidget(self.log)
        layout.addLayout(left, 3)

        # ---- Panneau de contrôle ----
        right = QtWidgets.QVBoxLayout()
        g = QtWidgets.QGridLayout(); r = 0

        # Caméra
        g.addWidget(QtWidgets.QLabel("<b>Caméra</b>"), r, 0, 1, 2); r += 1
        g.addWidget(QtWidgets.QLabel("Index"), r, 0); self.cam_index = QtWidgets.QSpinBox(); self.cam_index.setRange(0, 20); self.cam_index.setValue(5)
        g.addWidget(self.cam_index, r, 1); r += 1
        g.addWidget(QtWidgets.QLabel("Largeur"), r, 0); self.width_box = QtWidgets.QSpinBox(); self.width_box.setRange(320, 7680); self.width_box.setValue(3840)
        g.addWidget(self.width_box, r, 1); r += 1
        g.addWidget(QtWidgets.QLabel("Hauteur"), r, 0); self.height_box = QtWidgets.QSpinBox(); self.height_box.setRange(240, 4320); self.height_box.setValue(1080)
        g.addWidget(self.height_box, r, 1); r += 1
        g.addWidget(QtWidgets.QLabel("FPS"), r, 0); self.fps_box = QtWidgets.QDoubleSpinBox(); self.fps_box.setRange(1.0, 240.0); self.fps_box.setDecimals(2); self.fps_box.setValue(30.0)
        g.addWidget(self.fps_box, r, 1); r += 1

        # Damier
        g.addWidget(QtWidgets.QLabel("<b>Damier</b>"), r, 0, 1, 2); r += 1
        g.addWidget(QtWidgets.QLabel("Colonnes (int)"), r, 0); self.cols_box = QtWidgets.QSpinBox(); self.cols_box.setRange(3, 30); self.cols_box.setValue(9)
        g.addWidget(self.cols_box, r, 1); r += 1
        g.addWidget(QtWidgets.QLabel("Lignes (int)"), r, 0); self.rows_box = QtWidgets.QSpinBox(); self.rows_box.setRange(3, 30); self.rows_box.setValue(6)
        g.addWidget(self.rows_box, r, 1); r += 1
        g.addWidget(QtWidgets.QLabel("Taille carrés (m)"), r, 0); self.square_size = QtWidgets.QDoubleSpinBox(); self.square_size.setRange(0.001, 1.0); self.square_size.setDecimals(4); self.square_size.setValue(0.0200)
        g.addWidget(self.square_size, r, 1); r += 1

        # Capture
        g.addWidget(QtWidgets.QLabel("<b>Capture</b>"), r, 0, 1, 2); r += 1
        g.addWidget(QtWidgets.QLabel("Nombre de captures"), r, 0); self.n_captures = QtWidgets.QSpinBox(); self.n_captures.setRange(5, 200); self.n_captures.setValue(15)
        g.addWidget(self.n_captures, r, 1); r += 1
        g.addWidget(QtWidgets.QLabel("Auto-capture si détecté"), r, 0); self.auto_chk = QtWidgets.QCheckBox(); self.auto_chk.setChecked(False)
        g.addWidget(self.auto_chk, r, 1); r += 1
        g.addWidget(QtWidgets.QLabel("Délai min (s) auto"), r, 0); self.auto_delay = QtWidgets.QDoubleSpinBox(); self.auto_delay.setRange(0.0, 5.0); self.auto_delay.setDecimals(2); self.auto_delay.setValue(0.50)
        g.addWidget(self.auto_delay, r, 1); r += 1

        right.addLayout(g)

        # Boutons
        btns = QtWidgets.QGridLayout(); br = 0
        self.start_btn = QtWidgets.QPushButton("Démarrer caméra")
        self.stop_btn = QtWidgets.QPushButton("Arrêter caméra"); self.stop_btn.setEnabled(False)
        self.capture_btn = QtWidgets.QPushButton("Capturer paire (C)")
        self.reset_btn = QtWidgets.QPushButton("Réinitialiser")
        self.run_btn = QtWidgets.QPushButton("Lancer calibration")
        self.save_btn = QtWidgets.QPushButton("Enregistrer YAML…"); self.save_btn.setEnabled(False)

        btns.addWidget(self.start_btn, br, 0); btns.addWidget(self.stop_btn, br, 1); br += 1
        btns.addWidget(self.capture_btn, br, 0); btns.addWidget(self.reset_btn, br, 1); br += 1
        btns.addWidget(self.run_btn, br, 0); btns.addWidget(self.save_btn, br, 1); br += 1
        right.addLayout(btns)
        right.addStretch(1)
        layout.addLayout(right, 2)

        # Connexions
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.capture_btn.clicked.connect(self.on_capture_clicked)
        self.reset_btn.clicked.connect(self.on_reset)
        self.run_btn.clicked.connect(self.on_run_calibration)
        self.save_btn.clicked.connect(self.on_save_yaml)

        # Critères OpenCV
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Petit rappel d’aide
        self.log_append("Appuie sur 'C' pour capturer manuellement.")

    # ----- Journal -----
    def log_append(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {msg}")

    # ----- Caméra -----
    def on_start(self):
        if self.cap is not None:
            return
        idx = int(self.cam_index.value())
        w = int(self.width_box.value())
        h = int(self.height_box.value())
        fps = float(self.fps_box.value())
        self.cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            self.cap.release(); self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release(); self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if not self.cap or not self.cap.isOpened():
            self.log_append("ERREUR: impossible d’ouvrir la caméra.")
            self.cap = None
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
        self.cap.set(cv2.CAP_PROP_FPS, float(fps))
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        for _ in range(8):
            self.cap.grab()
        self.timer.start(0)
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.log_append(f"Caméra {idx} démarrée en {w}x{h}@{fps:.2f}.")

    def on_stop(self):
        self.timer.stop()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.log_append("Caméra arrêtée.")   

    # ----- Boucle de prévisualisation -----
    def _on_timer(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return
        w = int(self.width_box.value()) // 2
        left_img = frame[:, :w]
        right_img = frame[:, w:]
        gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        self.gray_size = gray_l.shape[::-1]

        chessboard_size = (int(self.cols_box.value()), int(self.rows_box.value()))
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)

        vis_l, vis_r = left_img.copy(), right_img.copy()
        if ret_l:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
            cv2.drawChessboardCorners(vis_l, chessboard_size, corners_l, ret_l)
        if ret_r:
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
            cv2.drawChessboardCorners(vis_r, chessboard_size, corners_r, ret_r)

        vis = cv2.hconcat([vis_l, vis_r])
        h, wtot = vis.shape[:2]
        qimg = QtGui.QImage(vis.data, wtot, h, vis.strides[0], QtGui.QImage.Format.Format_BGR888)
        self.preview.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        # Auto-capture si demandé
        if self.auto_chk.isChecked() and ret_l and ret_r:
            now_ms = int(time.time() * 1000)
            if now_ms - self.last_capture_ms >= int(self.auto_delay.value() * 1000):
                self._do_store_pair(corners_l, corners_r, chessboard_size)

        self._update_progress_label()

        # Mémoriser corners pour la capture manuelle
        self._last_detect = (ret_l, ret_r, corners_l if ret_l else None, corners_r if ret_r else None, chessboard_size)

    # ----- Capture manuelle -----
    def on_capture_clicked(self):
        if not hasattr(self, "_last_detect"):
            self.log_append("Aucun damier détecté encore.")
            return
        ret_l, ret_r, corners_l, corners_r, chessboard_size = self._last_detect
        if not (ret_l and ret_r):
            self.log_append("Damier non détecté dans les deux vues.")
            return
        self._do_store_pair(corners_l, corners_r, chessboard_size)

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() in (QtCore.Qt.Key_C, QtCore.Qt.Key_Space):
            self.on_capture_clicked()

    # ----- Stockage d'une paire -----
    def _do_store_pair(self, corners_l, corners_r, chessboard_size):
        # Objet 3D du damier
        objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= float(self.square_size.value())

        self.objpoints.append(objp.copy())
        self.imgpoints_left.append(corners_l)
        self.imgpoints_right.append(corners_r)
        self.last_capture_ms = int(time.time() * 1000)
        self._update_progress_label()
        self.log_append(f"Capture {len(self.objpoints)}/{int(self.n_captures.value())} enregistrée.")

        # Stop auto quand assez de captures
        if len(self.objpoints) >= int(self.n_captures.value()):
            if self.auto_chk.isChecked():
                self.auto_chk.setChecked(False)
                self.log_append("Auto-capture arrêtée (quota atteint).")

    def _update_progress_label(self):
        self.progress_lbl.setText(f"{len(self.objpoints)} capture(s) / {int(self.n_captures.value())}")

    # ----- Reset -----
    def on_reset(self):
        self.objpoints.clear()
        self.imgpoints_left.clear()
        self.imgpoints_right.clear()
        self.calib_data = None
        self.save_btn.setEnabled(False)
        self._update_progress_label()
        self.log_append("Captures réinitialisées.")

    # ----- Lancer la calibration -----
    def on_run_calibration(self):
        if len(self.objpoints) < 5:
            self.log_append("ERREUR: Trop peu de captures (min=5).")
            return
        if self.gray_size is None:
            self.log_append("ERREUR: Taille d'image inconnue.")
            return

        image_size = self.gray_size
        calib_flags = (cv2.CALIB_ZERO_TANGENT_DIST |
                       cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 |
                       cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)

        # Intrinsèques gauche / droite
        ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, image_size, None, None, flags=calib_flags
        )
        ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, image_size, None, None, flags=calib_flags
        )

        self.log_append(f"RMS gauche: {ret_l:.4f}  |  RMS droite: {ret_r:.4f}")

        # Stéréo (intrinsèques fixées)
        stereo_flags = calib_flags | cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_left, self.imgpoints_right,
            mtx_l, dist_l, mtx_r, dist_r, image_size, criteria=criteria, flags=stereo_flags
        )
        self.log_append(f"RMS stéréo: {ret_stereo:.4f}")
        self.log_append(f"T = {T.ravel()}")

        # Rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx_l, dist_l, mtx_r, dist_r, image_size, R, T, alpha=0
        )

        self.calib_data = {
            'camera_matrix_left': mtx_l.tolist(),
            'dist_coeff_left': dist_l.tolist(),
            'camera_matrix_right': mtx_r.tolist(),
            'dist_coeff_right': dist_r.tolist(),
            'R': R.tolist(),
            'T': T.tolist(),
            'E': E.tolist(),
            'F': F.tolist(),
            'R1': R1.tolist(),
            'R2': R2.tolist(),
            'P1': P1.tolist(),
            'P2': P2.tolist(),
            'Q': Q.tolist(),
            'rms_left': float(ret_l),
            'rms_right': float(ret_r),
            'rms_stereo': float(ret_stereo),
            'image_width': int(image_size[0]),
            'image_height': int(image_size[1]),
            'chessboard_cols': int(self.cols_box.value()),
            'chessboard_rows': int(self.rows_box.value()),
            'square_size_m': float(self.square_size.value()),
            'n_captures': int(len(self.objpoints)),
            'timestamp': datetime.now().isoformat(timespec="seconds")
        }
        self.save_btn.setEnabled(True)
        self.log_append("Calibration OK — prêt à enregistrer le YAML.")

    # ----- Sauvegarde YAML -----
    def on_save_yaml(self):
        if self.calib_data is None:
            self.log_append("Rien à enregistrer — lance la calibration d’abord.")
            return
        default_name = f"stereo_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Enregistrer la calibration YAML", default_name, "YAML (*.yaml)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(self.calib_data, f)
            self.log_append(f"Calibration enregistrée → {Path(path).name}")
        except Exception as e:
            self.log_append(f"ERREUR écriture YAML: {e}")


# ============================ CaptureCam Page ============================ #
class CaptureCamPage(QtWidgets.QWidget):
    status_msg = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.output_dir = Path.cwd() / "captures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._preview_counter = 0
        self._do_snapshot = False
        self.metadata_json_path = None
        self.video_path = None

        # --- Layout principal ---
        layout = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()
        self.preview = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(960, 540)
        # Petit logo en overlay (coin haut-gauche)
        self.logo_label = QtWidgets.QLabel(self.preview)
        self.logo_label.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.logo_label.setStyleSheet("background: transparent;")
        try:
            _lp = find_logo_path()
            if _lp:
                _pm = QtGui.QPixmap(str(_lp))
                if not _pm.isNull():
                    self.logo_label.setPixmap(_pm.scaledToWidth(120, QtCore.Qt.SmoothTransformation))
                    self.logo_label.move(10, 10)
                    self.logo_label.show()
                else:
                    self.logo_label.hide()
        except Exception:
            self.logo_label.hide()
        self.preview.setStyleSheet("background:#111; color:#aaa; border:1px solid #333;")
        left.addWidget(self.preview, 1)
        self.status_line = QtWidgets.QLabel("Idle."); self.status_line.setStyleSheet("color:#0f0")
        left.addWidget(self.status_line)
        layout.addLayout(left, 3)

        right = QtWidgets.QVBoxLayout()
        grid = QtWidgets.QGridLayout(); row = 0
        grid.addWidget(QtWidgets.QLabel("Index caméra"), row, 0)
        self.cam_index = QtWidgets.QSpinBox(); self.cam_index.setRange(0, 20); self.cam_index.setValue(0)
        grid.addWidget(self.cam_index, row, 1); row += 1
        grid.addWidget(QtWidgets.QLabel("Largeur"), row, 0)
        self.width_box = QtWidgets.QSpinBox(); self.width_box.setRange(160, 7680); self.width_box.setValue(3840)
        grid.addWidget(self.width_box, row, 1); row += 1
        grid.addWidget(QtWidgets.QLabel("Hauteur"), row, 0)
        self.height_box = QtWidgets.QSpinBox(); self.height_box.setRange(120, 4320); self.height_box.setValue(1080)
        grid.addWidget(self.height_box, row, 1); row += 1
        grid.addWidget(QtWidgets.QLabel("FPS"), row, 0)
        self.fps_box = QtWidgets.QDoubleSpinBox(); self.fps_box.setRange(1.0, 240.0); self.fps_box.setDecimals(2); self.fps_box.setValue(60.0)
        grid.addWidget(self.fps_box, row, 1); row += 1
        self.burnin_chk = QtWidgets.QCheckBox("Burn‑in (TC + UTC) sur l’aperçu"); self.burnin_chk.setChecked(True)
        grid.addWidget(self.burnin_chk, row, 0, 1, 2); row += 1
        self.preview_skip_label = QtWidgets.QLabel("Décimation d’aperçu (afficher 1 image sur N)")
        self.preview_skip = QtWidgets.QSpinBox(); self.preview_skip.setRange(1, 8); self.preview_skip.setValue(1)
        grid.addWidget(self.preview_skip_label, row, 0); grid.addWidget(self.preview_skip, row, 1); row += 1
        right.addLayout(grid)

        out_box = QtWidgets.QGroupBox("Sortie")
        out_lay = QtWidgets.QGridLayout(out_box)
        out_lay.addWidget(QtWidgets.QLabel("Dossier"), 0, 0)
        self.out_dir_label = QtWidgets.QLineEdit(str(self.output_dir)); self.out_dir_label.setReadOnly(True)
        out_lay.addWidget(self.out_dir_label, 0, 1)
        browse_btn = QtWidgets.QPushButton("Parcourir…"); browse_btn.clicked.connect(self.choose_output_dir)
        out_lay.addWidget(browse_btn, 0, 2)
        out_lay.addWidget(QtWidgets.QLabel("Nom de base"), 1, 0)
        self.base_name_edit = QtWidgets.QLineEdit("session")
        out_lay.addWidget(self.base_name_edit, 1, 1, 1, 2)
        right.addWidget(out_box)

        # Métadonnées
        self.meta_panel = MetadataPanel()
        right.addWidget(self.meta_panel)

        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Démarrer caméra")
        self.stop_btn = QtWidgets.QPushButton("Arrêter caméra")
        self.rec_btn = QtWidgets.QPushButton("● REC")
        self.snap_btn = QtWidgets.QPushButton("Snapshot")
        for b in (self.stop_btn, self.rec_btn): b.setEnabled(False)
        btn_layout.addWidget(self.start_btn); btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.rec_btn); btn_layout.addWidget(self.snap_btn)
        right.addLayout(btn_layout)
        right.addStretch(1)
        layout.addLayout(right, 2)

        # Threads capture & writer
        self.capture_worker = CaptureWorker(); self.capture_thread = QtCore.QThread(self)
        self.capture_worker.moveToThread(self.capture_thread); self.capture_thread.start()
        self.writer_worker = VideoWriterWorker(); self.writer_thread = QtCore.QThread(self)
        self.writer_worker.moveToThread(self.writer_thread); self.writer_thread.start()
        self.capture_worker.attach_writer(self.writer_worker)

        # Connexions
        self.start_btn.clicked.connect(self.on_start_cam)
        self.stop_btn.clicked.connect(self.on_stop_cam)
        self.rec_btn.clicked.connect(self.on_toggle_record)
        self.snap_btn.clicked.connect(self.on_snapshot)
        self.capture_worker.frame_ready.connect(self.on_frame)
        self.capture_worker.status_msg.connect(self.status_line.setText)
        self.writer_worker.status_msg.connect(self.status_line.setText)

        # états
        self.recording = False

    # ----- Handlers & helpers ----- #
    def choose_output_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie", str(self.output_dir))
        if d:
            self.output_dir = Path(d); self.out_dir_label.setText(str(self.output_dir))

    def on_start_cam(self):
        self.capture_worker.configure(
            device_index=int(self.cam_index.value()),
            width=int(self.width_box.value()),
            height=int(self.height_box.value()),
            fps=float(self.fps_box.value()),
            burnin=bool(self.burnin_chk.isChecked()),
        )
        QtCore.QMetaObject.invokeMethod(self.capture_worker, "start")
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True); self.rec_btn.setEnabled(True)

    def on_stop_cam(self):
        if self.recording:
            self.on_toggle_record()
        QtCore.QMetaObject.invokeMethod(self.capture_worker, "stop")
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False); self.rec_btn.setEnabled(False)

    def _make_output_paths(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{self.base_name_edit.text().strip() or 'session'}_{ts}"
        video_base = self.output_dir / base
        csv_path = self.output_dir / f"{base}_frames.csv"
        json_path = self.output_dir / f"{base}_metadata.json"
        snap_dir = self.output_dir / f"{base}_snaps"; snap_dir.mkdir(parents=True, exist_ok=True)
        return video_base, csv_path, json_path, snap_dir

    def on_toggle_record(self):
        if not self.recording:
            video_base, csv_path, json_path, snap_dir = self._make_output_paths()
            self.snap_dir = snap_dir
            self._pending_paths = (video_base, csv_path, json_path)
            self.recording = True
            self.rec_btn.setText("■ STOP")
            self.status_line.setText("Enregistrement: initialisation…")
        else:
            self.recording = False
            self.rec_btn.setText("● REC")
            self.capture_worker.recording = False
            QtCore.QMetaObject.invokeMethod(self.writer_worker, 'stop')
            try:
                if hasattr(self, 'writer_pythread') and self.writer_pythread is not None:
                    self.writer_pythread.join(timeout=5.0)
                    self.writer_pythread = None
            except Exception:
                pass
            self.status_line.setText("Enregistrement arrêté.")

    def on_snapshot(self):
        if not hasattr(self, "snap_dir"):
            self.snap_dir = self.output_dir / "snaps"
        self._do_snapshot = True
        self.status_line.setText("Snapshot armé — prochaine frame.")

    def on_frame(self, frame: np.ndarray, ts_posix: float, mono_ns: int, tc_str: str):
        # Aperçu décimé
        self._preview_counter = (self._preview_counter + 1) % int(max(1, self.preview_skip.value()))
        if self._preview_counter == 0:
            h, w = frame.shape[:2]
            qimg = QtGui.QImage(frame.data, w, h, frame.strides[0], QtGui.QImage.Format.Format_BGR888)
            pix = QtGui.QPixmap.fromImage(qimg)
            self.preview.setPixmap(pix.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        # Initialiser writer sur 1ère frame
        if self.recording and not self.capture_worker.recording and hasattr(self, "_pending_paths"):
            video_base, csv_path, json_path = self._pending_paths
            if not self.writer_worker.open(video_base, frame.shape[1], frame.shape[0], float(self.fps_box.value())):
                self.status_line.setText("ERREUR: impossible d’ouvrir le VideoWriter.")
                self.recording = False
                self.rec_btn.setText("● REC")
                return
            # CSV
            self.writer_worker.csv_file = open(csv_path, "w", newline='', encoding="utf-8")
            self.writer_worker.csv_writer = csv.writer(self.writer_worker.csv_file)
            self.writer_worker.csv_writer.writerow(["frame_idx", "dropped", "tc_smpte", "utc_iso", "local_iso", "monotonic_ns", "width", "height"])
            # JSON métadonnées
            self.meta_panel.set_path_video(str(self.writer_worker.video_path))
            metadata = self.meta_panel.to_dict()
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(metadata, jf, indent=2, ensure_ascii=False)
            # Lancer writer loop
            self.writer_worker._stop = False
            self.writer_worker._written_frames = 0
            self.writer_worker._dropped_count = 0
            self.writer_pythread = threading.Thread(target=self.writer_worker.loop, daemon=False)
            self.writer_pythread.start()
            # Activer file côté capture
            self.capture_worker.recording = True
            del self._pending_paths
            self.status_line.setText(f"Save → {self.writer_worker.video_path.name} (codec {self.writer_worker.fourcc_used}).")

        # Snapshot
        if getattr(self, "_do_snapshot", False):
            self._do_snapshot = False
            snap_dir = getattr(self, "snap_dir", self.output_dir / "snaps"); snap_dir.mkdir(parents=True, exist_ok=True)
            fn = snap_dir / (datetime.now().strftime("snap_%Y%m%d_%H%M%S_%f")[:-3] + ".png")
            cv2.imwrite(str(fn), frame)
            self.status_line.setText(f"Snapshot sauvegardé: {fn.name}")

    def shutdown(self):
        if getattr(self, 'recording', False):
            self.on_toggle_record()
        QtCore.QMetaObject.invokeMethod(self.capture_worker, 'stop')
        try:
            if hasattr(self, 'writer_pythread') and self.writer_pythread is not None:
                self.writer_pythread.join(timeout=5.0)
                self.writer_pythread = None
        except Exception:
            pass


# ============================ Pages Placeholder ============================ #
class PlaceholderPage(QtWidgets.QWidget):
    def __init__(self, title: str, subtitle: str = "Module à brancher…", parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lbl = QtWidgets.QLabel(f"<h2 style='margin:0'>{title}</h2><div>{subtitle}</div>")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lay.addStretch(1); lay.addWidget(lbl); lay.addStretch(1)


# ============================ Wrappers ============================ #
class VDAWrapper:
    def __init__(self, encoder: str = "vits"):
        self.encoder = encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_configs = {
            "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        if encoder not in model_configs:
            raise ValueError(f"Unknown encoder={encoder}. Use 'vits' or 'vitl'.")

        ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", f"video_depth_anything_{encoder}.pth")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        self.model = VideoDepthAnything(**model_configs[encoder])
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def infer_depth_colormap_bgr(self, frame_bgr_u8: np.ndarray, input_size: int = 518) -> np.ndarray:
        """
        Input: BGR uint8
        Output: BGR uint8 colored depth map (OpenCV colormap)
        """
        rgb = cv2.cvtColor(frame_bgr_u8, cv2.COLOR_BGR2RGB)
        depth = self.model.infer_video_depth_one(rgb, input_size=input_size, device=self.device)

        # depth -> numpy
        depth = depth.squeeze()
        depth_np = depth.detach().cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)

        # robust normalize
        lo = np.percentile(depth_np, 2.0)
        hi = np.percentile(depth_np, 98.0)
        depth_norm = (depth_np - lo) / (hi - lo + 1e-6)
        depth_norm = np.clip(depth_norm, 0.0, 1.0)

        depth_u8 = (depth_norm * 255).astype(np.uint8)
        depth_col = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)  # BGR
        return depth_col


# ============================ Application MainWindow ============================ #
class AppWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — Capture / Calibrate / Pose")
        # Window icon = logo if available
        try:
            _lp = find_logo_path()
            if _lp:
                self.setWindowIcon(QtGui.QIcon(str(_lp)))
        except Exception:
            pass
        self.resize(1500, 900)

        # Stacked pages
        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.capture_page = CaptureCamPage(self)
        self.page_calib_mono = PlaceholderPage("Calibrate Mono", "Ici on chargera votre module de calibration mono.")
        self.page_calib_stereo = CalibrateStereoPage(self)  # <<< intégré
        self.page_live_pose = LivePosePage(self)
        self.page_pose_graph = PlaceholderPage("Pose Graph", "Visualisation du graphe de poses.")
        self.page_exr = ExrSequencePage(self)
        self.vda_wrapper = build_vda_wrapper()
        self.page_exr = ExrSequencePage(vda_wrapper=self.vda_wrapper)


        for p in (self.capture_page, self.page_calib_mono, self.page_calib_stereo, self.page_live_pose, self.page_pose_graph, self.page_exr):
            self.stack.addWidget(p)
        self.stack.setCurrentWidget(self.capture_page)

        # Menus
        self._build_menus()

    def _build_menus(self):
        mb = self.menuBar()
        # File
        m_file = mb.addMenu("File")
        act_open = QtWidgets.QAction("Open Session", self)
        act_capture = QtWidgets.QAction("Capture Cam", self)
        act_save = QtWidgets.QAction("Save Session", self)
        m_file.addAction(act_open); m_file.addAction(act_capture); m_file.addAction(act_save)
        act_open.triggered.connect(self.on_open_session)
        act_capture.triggered.connect(lambda: self.stack.setCurrentWidget(self.capture_page))
        act_save.triggered.connect(self.on_save_session)
        act_open_exr = QtWidgets.QAction("Open EXR Sequence…", self)
        m_file.addAction(act_open_exr)
        act_open_exr.triggered.connect(self.on_open_exr_sequence)

        # Calibrate
        m_cal = mb.addMenu("Calibrate")
        act_cmono = QtWidgets.QAction("Calibrate Mono", self)
        act_cstereo = QtWidgets.QAction("Calibrate Stereo", self)
        m_cal.addAction(act_cmono); m_cal.addAction(act_cstereo)
        act_cmono.triggered.connect(lambda: self.stack.setCurrentWidget(self.page_calib_mono))
        act_cstereo.triggered.connect(lambda: self.stack.setCurrentWidget(self.page_calib_stereo))
        # Pose Estimation
        m_pose = mb.addMenu("Pose Estimation")
        act_live = QtWidgets.QAction("Live Pose", self)
        act_graph = QtWidgets.QAction("Pose Graph", self)
        m_pose.addAction(act_live); m_pose.addAction(act_graph)
        act_live.triggered.connect(lambda: self.stack.setCurrentWidget(self.page_live_pose))
        act_graph.triggered.connect(lambda: self.stack.setCurrentWidget(self.page_pose_graph))


    # ----- File actions ----- #
    def on_open_exr_sequence(self):
        self.stack.setCurrentWidget(self.page_exr)
        self.page_exr.open_exr_dir()


    def on_open_session(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Ouvrir le JSON de session", str(Path.cwd() / "captures"), "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.capture_page.meta_panel.load_from_dict(data)
            self.statusBar().showMessage(f"Session chargée depuis {Path(path).name}", 5000)
            self.stack.setCurrentWidget(self.capture_page)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur", f"Impossible de charger le JSON:\n{e}")

    def on_save_session(self):
        data = self.capture_page.meta_panel.to_dict()
        if not data.get('path_video') and self.capture_page.video_path:
            data['path_video'] = self.capture_page.video_path
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Enregistrer la session JSON", str(Path.cwd() / "captures" / "session.json"), "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.statusBar().showMessage(f"Session sauvegardée → {Path(path).name}", 5000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur", f"Impossible d'enregistrer le JSON:\n{e}")

    # graceful close
    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.capture_page.shutdown()
        except Exception:
            pass
        event.accept()


# ============================ Entrée ============================ #
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = AppWindow(); w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
