import os
import json
import time
import queue
import threading
import traceback
from collections import deque, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None

import torch
import torch.nn as nn

import tkinter as tk
from tkinter import ttk, messagebox


# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "resnet1d_em_fingerprint_final.pth")
SCRIPTED_MODEL_PATH = os.path.join(BASE_DIR, "resnet1d_em_fingerprint_scripted.pt")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")
META_PATH = os.path.join(BASE_DIR, "training_metadata.json")


# DEFAULT CONFIG
DEFAULT_START_HZ = 2_000_000_000
DEFAULT_STOP_HZ = 3_000_000_000
DEFAULT_POINTS = 401
DEFAULT_INTERVAL_S = 0.5
DEFAULT_BAUDRATE = 115200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE_THRESHOLD = 0.75
VOTE_WINDOW = 5
VOTE_MIN_COUNT = 4
KEEP_LAST_STABLE = True

# These must match training.
ALIGN_TO_CENTER = True
USE_UNWRAP_PHASE = True
USE_ZSCORE = True


# MODEL
class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, k: int = 7):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        return torch.relu(out + identity)


class ResNet1D(nn.Module):
    def __init__(self, n_classes: int, in_ch: int = 4, emb_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResBlock1D(64, 64, stride=1, k=7),
            ResBlock1D(64, 64, stride=1, k=7),
        )
        self.layer2 = nn.Sequential(
            ResBlock1D(64, 128, stride=2, k=7),
            ResBlock1D(128, 128, stride=1, k=7),
        )
        self.layer3 = nn.Sequential(
            ResBlock1D(128, 256, stride=2, k=5),
            ResBlock1D(256, 256, stride=1, k=5),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_emb = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, emb_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.35)
        self.fc_out = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        emb = self.fc_emb(x)
        logits = self.fc_out(self.dropout(emb))
        return logits


# PREPROCESSING
def first_diff(x: np.ndarray) -> np.ndarray:
    d = np.diff(x, axis=1)
    d = np.concatenate([d, np.zeros((x.shape[0], 1), dtype=np.float32)], axis=1)
    return d.astype(np.float32)


def per_sample_channel_zscore(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True)
    return ((X - mean) / (std + eps)).astype(np.float32)


def align_by_mag_min_single(mag: np.ndarray, phase: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = mag.shape[1] // 2
    idx = int(np.argmin(mag[0]))
    shift = center - idx
    return np.roll(mag, shift, axis=1), np.roll(phase, shift, axis=1)


def preprocess_re_im(re: np.ndarray, im: np.ndarray, expected_points: Optional[int] = None) -> np.ndarray:
    re = np.asarray(re, dtype=np.float32).reshape(1, -1)
    im = np.asarray(im, dtype=np.float32).reshape(1, -1)

    if expected_points is not None and re.shape[1] != expected_points:
        old_x = np.linspace(0.0, 1.0, re.shape[1])
        new_x = np.linspace(0.0, 1.0, expected_points)
        re = np.interp(new_x, old_x, re[0]).astype(np.float32).reshape(1, -1)
        im = np.interp(new_x, old_x, im[0]).astype(np.float32).reshape(1, -1)

    mag = np.sqrt(re * re + im * im).astype(np.float32)
    phase = np.arctan2(im, re).astype(np.float32)

    if USE_UNWRAP_PHASE:
        phase = np.unwrap(phase, axis=1).astype(np.float32)
    if ALIGN_TO_CENTER:
        mag, phase = align_by_mag_min_single(mag, phase)

    dmag = first_diff(mag)
    dphase = first_diff(phase)
    X = np.stack([mag[0], phase[0], dmag[0], dphase[0]], axis=0).astype(np.float32)
    X = X.reshape(1, 4, X.shape[1])

    if USE_ZSCORE:
        X = per_sample_channel_zscore(X)
    return X.astype(np.float32)


# FILE LOADING
def load_json_file(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {os.path.basename(path)} in:\n{BASE_DIR}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label_names(path: str) -> List[str]:
    data = load_json_file(path)

    if all(not isinstance(v, dict) for v in data.values()) and "label_map" not in data and "inv_map" not in data:
        label_map = {str(k): int(v) for k, v in data.items()}
        return [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    if "inv_map" in data and isinstance(data["inv_map"], dict):
        inv_map = data["inv_map"]
        return [str(inv_map[str(i)]) for i in sorted([int(k) for k in inv_map.keys()])]

    if "label_map" in data and isinstance(data["label_map"], dict):
        label_map = {str(k): int(v) for k, v in data["label_map"].items()}
        return [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    raise ValueError("Unsupported label_map.json format.")


def load_metadata(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        return load_json_file(path)
    except Exception:
        return {}


def find_state_dict(obj):
    if isinstance(obj, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if any(isinstance(k, str) and (k.startswith("stem.") or k.startswith("layer") or k.startswith("fc_out")) for k in obj.keys()):
            return obj
    raise ValueError("Cannot find model state_dict in the checkpoint file.")


def load_model_and_labels():
    label_names = load_label_names(LABEL_MAP_PATH)
    metadata = load_metadata(META_PATH)

    expected_points = None
    for key in ["input_length", "points", "n_points", "num_points", "expected_points"]:
        if key in metadata:
            try:
                expected_points = int(metadata[key])
                break
            except Exception:
                pass
    if expected_points is None:
        expected_points = DEFAULT_POINTS

    n_classes = len(label_names)

    if os.path.exists(SCRIPTED_MODEL_PATH):
        model = torch.jit.load(SCRIPTED_MODEL_PATH, map_location=DEVICE)
        model.eval()
        return model, label_names, expected_points, metadata

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Cannot find {os.path.basename(MODEL_PATH)} in:\n{BASE_DIR}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = find_state_dict(checkpoint)

    model = ResNet1D(n_classes=n_classes, in_ch=4, emb_dim=128).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, label_names, expected_points, metadata


# NANOVNA SERIAL
@dataclass
class VNAConfig:
    port: str
    baudrate: int
    start_hz: int
    stop_hz: int
    points: int
    interval_s: float


class NanoVNAClient:
    def __init__(self, config: VNAConfig):
        if serial is None:
            raise RuntimeError("pyserial is not installed. Run: pip install pyserial")
        self.config = config
        self.ser = None
        self.last_raw_lines: List[str] = []

    def connect(self):
        self.ser = serial.Serial(self.config.port, baudrate=self.config.baudrate, timeout=0.2, write_timeout=1)
        time.sleep(0.6)
        self._flush()

        self._write_raw("\r\n")
        time.sleep(0.2)
        self._drain(timeout_s=0.5)

        sweep_cmds = [
            f"sweep {self.config.start_hz} {self.config.stop_hz} {self.config.points}",
            f"sweep {self.config.start_hz} {self.config.stop_hz}",
        ]
        for cmd in sweep_cmds:
            self._send(cmd)
            time.sleep(0.25)
            self._drain(timeout_s=0.8)

    def close(self):
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    def _flush(self):
        if self.ser is not None:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

    def _write_raw(self, text: str):
        if self.ser is None:
            raise RuntimeError("Serial port is not connected.")
        self.ser.write(text.encode("ascii", errors="ignore"))

    def _send(self, cmd: str):
        # Use CRLF for better compatibility with NanoVNA-F / H variants.
        self._write_raw(cmd.strip() + "\r\n")

    def _drain(self, timeout_s: float = 0.8) -> List[str]:
        if self.ser is None:
            return []
        lines = []
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            if line:
                lines.append(line)
        return lines

    def _read_lines(self, timeout_s: float = 6.0, expected_count: Optional[int] = None) -> List[str]:
        if self.ser is None:
            raise RuntimeError("Serial port is not connected.")
        lines = []
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            if not line:
                continue
            # NanoVNA shells often end with ch> prompt.
            if line.startswith("ch>") or line == ">":
                if expected_count is None or len(lines) >= expected_count:
                    break
                continue
            lines.append(line)
            if expected_count is not None and len(lines) >= expected_count:
                break
        return lines

    @staticmethod
    def _parse_complex_lines(lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        re_vals, im_vals = [], []
        for line in lines:
            text = line.strip()
            if not text:
                continue
            low = text.lower()
            # Skip prompts, command echoes and text messages.
            if low.startswith(("ch>", ">", "data", "sweep", "frequ", "version", "nano", "error", "usage", "cmd")):
                continue
            # Accept formats like: "0.123 -0.456", "0.123,-0.456", "0.123\t-0.456"
            parts = text.replace(",", " ").replace(";", " ").split()
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass
            if len(nums) >= 2:
                re_vals.append(nums[0])
                im_vals.append(nums[1])
        return np.asarray(re_vals, dtype=np.float32), np.asarray(im_vals, dtype=np.float32)

    def query_raw(self, cmd: str, timeout_s: float = 3.0) -> List[str]:
        if self.ser is None:
            raise RuntimeError("Serial port is not connected.")
        self._flush()
        self._send(cmd)
        lines = self._read_lines(timeout_s=timeout_s)
        self.last_raw_lines = lines
        return lines

    def read_s11(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read S11 complex trace. Returns re, im arrays."""
        if self.ser is None:
            raise RuntimeError("Serial port is not connected.")

        raw_collected = []
        candidate_cmds = ["data 0", "data", "data 1"]

        for cmd in candidate_cmds:
            self._flush()
            self._send(cmd)
            lines = self._read_lines(timeout_s=6.0, expected_count=self.config.points)
            raw_collected.extend([f"[{cmd}] {ln}" for ln in lines[:12]])
            re, im = self._parse_complex_lines(lines)
            if len(re) >= max(20, min(50, self.config.points // 3)):
                self.last_raw_lines = lines
                return re, im

        self.last_raw_lines = raw_collected
        preview = "\n".join(raw_collected[:10]) if raw_collected else "No text returned from the serial port."
        raise RuntimeError(
            "No valid VNA data received.\n\n"
            "Most likely causes:\n"
            "1) Wrong COM port selected.\n"
            "2) NanoVNA is not in USB serial shell mode.\n"
            "3) Another program is using the port.\n"
            "4) This NanoVNA firmware uses a different serial command.\n\n"
            f"Raw response preview:\n{preview}"
        )


# APP
class EMFingerprintApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Electromagnetic Identification")
        self.root.geometry("940x640")
        self.root.minsize(900, 600)
        self.root.configure(bg="#0B1220")

        self.model = None
        self.label_names = []
        self.expected_points = DEFAULT_POINTS
        self.metadata = {}

        self.worker_thread = None
        self.stop_event = threading.Event()
        self.msg_queue = queue.Queue()
        self.vna = None

        # Stable-result filter state
        self.prediction_buffer = deque(maxlen=VOTE_WINDOW)
        self.confidence_buffer = deque(maxlen=VOTE_WINDOW)
        self.stable_prediction = None
        self.stable_confidence = None

        self._build_style()
        self._build_ui()
        self._load_model_safely()
        self._refresh_ports()
        self._poll_queue()

    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#0B1220")
        style.configure("Card.TFrame", background="#121C2E")
        style.configure("TLabel", background="#0B1220", foreground="#DDE7FF", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#121C2E", foreground="#DDE7FF", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#0B1220", foreground="#FFFFFF", font=("Segoe UI", 24, "bold"))
        style.configure("Subtitle.TLabel", background="#0B1220", foreground="#AFC3E8", font=("Segoe UI", 10))
        style.configure("Prediction.TLabel", background="#121C2E", foreground="#54D7FF", font=("Segoe UI", 42, "bold"))
        style.configure("Confidence.TLabel", background="#121C2E", foreground="#FFFFFF", font=("Segoe UI", 17))
        style.configure("Status.TLabel", background="#0B1220", foreground="#AFC3E8", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=10)
        style.configure("Danger.TButton", font=("Segoe UI", 11, "bold"), padding=10)
        style.configure("TCombobox", padding=5)
        style.configure("TEntry", padding=5)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=24)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Real-Time Electromagnetic Identification", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="NanoVNA serial streaming  •  ResNet1D inference  •  Shape classification",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(6, 22))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)

        result_card = ttk.Frame(body, style="Card.TFrame", padding=22)
        result_card.pack(side="left", fill="both", expand=True, padx=(0, 14))

        ttk.Label(result_card, text="Current Prediction", style="Card.TLabel", font=("Segoe UI", 12)).pack(anchor="w")
        self.prediction_label = ttk.Label(result_card, text="Waiting", style="Prediction.TLabel")
        self.prediction_label.pack(anchor="w", pady=(24, 4))

        self.confidence_label = ttk.Label(result_card, text="Confidence: --", style="Confidence.TLabel")
        self.confidence_label.pack(anchor="w", pady=(0, 8))

        self.decision_status_label = ttk.Label(
            result_card,
            text="Decision Status: Waiting",
            style="Card.TLabel",
            font=("Segoe UI", 12)
        )
        self.decision_status_label.pack(anchor="w", pady=(0, 24))

        top_card = ttk.Frame(result_card, style="Card.TFrame", padding=14)
        top_card.pack(fill="x", pady=(8, 0))
        ttk.Label(top_card, text="Top Predictions", style="Card.TLabel", font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 10))
        self.top1_label = ttk.Label(top_card, text="#1  --", style="Card.TLabel", font=("Segoe UI", 12))
        self.top2_label = ttk.Label(top_card, text="#2  --", style="Card.TLabel", font=("Segoe UI", 12))
        self.top3_label = ttk.Label(top_card, text="#3  --", style="Card.TLabel", font=("Segoe UI", 12))
        self.top1_label.pack(anchor="w", pady=3)
        self.top2_label.pack(anchor="w", pady=3)
        self.top3_label.pack(anchor="w", pady=3)

        self.model_status_label = ttk.Label(result_card, text="Model: not loaded", style="Card.TLabel", font=("Segoe UI", 9))
        self.model_status_label.pack(anchor="w", pady=(28, 0))

        control_card = ttk.Frame(body, style="Card.TFrame", padding=22)
        control_card.pack(side="right", fill="y")

        ttk.Label(control_card, text="Connection", style="Card.TLabel", font=("Segoe UI", 17, "bold")).pack(anchor="w", pady=(0, 16))

        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(control_card, textvariable=self.port_var, width=28, state="readonly")
        self.port_combo.pack(anchor="w", pady=(0, 10))

        ttk.Button(control_card, text="Refresh Ports", command=self._refresh_ports).pack(anchor="w", pady=(0, 14))
        ttk.Button(control_card, text="Test VNA", command=self._test_vna).pack(anchor="w", pady=(0, 18))

        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUDRATE))
        self.start_var = tk.StringVar(value=str(DEFAULT_START_HZ))
        self.stop_var = tk.StringVar(value=str(DEFAULT_STOP_HZ))
        self.points_var = tk.StringVar(value=str(DEFAULT_POINTS))
        self.interval_var = tk.StringVar(value=str(DEFAULT_INTERVAL_S))

        self._entry_row(control_card, "Baud", self.baud_var)
        self._entry_row(control_card, "Start Hz", self.start_var)
        self._entry_row(control_card, "Stop Hz", self.stop_var)
        self._entry_row(control_card, "Points", self.points_var)
        self._entry_row(control_card, "Interval s", self.interval_var)

        self.start_button = ttk.Button(control_card, text="Start Recognition", style="Accent.TButton", command=self._start)
        self.start_button.pack(fill="x", pady=(24, 8))

        self.stop_button = ttk.Button(control_card, text="Stop", style="Danger.TButton", command=self._stop, state="disabled")
        self.stop_button.pack(fill="x")

        self.status_label = ttk.Label(outer, text="Status: Ready", style="Status.TLabel")
        self.status_label.pack(anchor="w", pady=(16, 0))

    def _entry_row(self, parent, label: str, var: tk.StringVar):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=5)
        ttk.Label(row, text=label, style="Card.TLabel", width=10).pack(side="left")
        ttk.Entry(row, textvariable=var, width=16).pack(side="left")

    def _load_model_safely(self):
        try:
            self.model, self.label_names, self.expected_points, self.metadata = load_model_and_labels()
            self.model_status_label.configure(
                text=f"Model loaded: {len(self.label_names)} classes | Device: {DEVICE} | Folder: {BASE_DIR}"
            )
            self.status_label.configure(text="Status: Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Model Loading Error", str(e))
            self.model_status_label.configure(text="Model loading failed")
            self.status_label.configure(text="Status: Model loading failed")

    def _refresh_ports(self):
        if serial is None:
            self.port_combo["values"] = []
            self.status_label.configure(text="Status: pyserial is not installed. Run: pip install pyserial")
            return
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])
        if ports:
            self.status_label.configure(text=f"Status: Found {len(ports)} serial port(s)")
        else:
            self.status_label.configure(text="Status: No serial ports found")

    def _read_config(self) -> VNAConfig:
        port = self.port_var.get().strip()
        if not port:
            raise ValueError("Please select a serial port.")
        return VNAConfig(
            port=port,
            baudrate=int(float(self.baud_var.get())),
            start_hz=int(float(self.start_var.get())),
            stop_hz=int(float(self.stop_var.get())),
            points=int(float(self.points_var.get())),
            interval_s=float(self.interval_var.get()),
        )

    def _test_vna(self):
        try:
            config = self._read_config()
            self.status_label.configure(text="Status: Testing NanoVNA connection...")
            t = threading.Thread(target=self._test_vna_worker, args=(config,), daemon=True)
            t.start()
        except Exception as e:
            messagebox.showerror("Configuration Error", str(e))

    def _test_vna_worker(self, config: VNAConfig):
        vna = None
        try:
            vna = NanoVNAClient(config)
            vna.connect()
            version_lines = vna.query_raw("version", timeout_s=2.0)
            re, im = vna.read_s11()
            msg = (
                f"VNA test passed. Received {len(re)} complex points. "
                f"Version response: {' | '.join(version_lines[:3]) if version_lines else 'No version text'}"
            )
            self.msg_queue.put(("status", "Status: " + msg))
            self.msg_queue.put(("info", msg))
        except Exception as e:
            self.msg_queue.put(("error", f"VNA test failed: {e}"))
            self.msg_queue.put(("debug", traceback.format_exc()))
        finally:
            if vna is not None:
                vna.close()

    def _start(self):
        if self.model is None:
            messagebox.showerror("Model Error", "Model is not loaded.")
            return
        try:
            config = self._read_config()
        except Exception as e:
            messagebox.showerror("Configuration Error", str(e))
            return

        self.stop_event.clear()
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
        self.stable_prediction = None
        self.stable_confidence = None
        self.prediction_label.configure(text="Waiting")
        self.confidence_label.configure(text="Confidence: --")
        self.decision_status_label.configure(text="Decision Status: Collecting")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_label.configure(text="Status: Connecting to NanoVNA...")

        self.worker_thread = threading.Thread(target=self._worker_loop, args=(config,), daemon=True)
        self.worker_thread.start()

    def _stop(self):
        self.stop_event.set()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_label.configure(text="Status: Stopping...")

    def _worker_loop(self, config: VNAConfig):
        try:
            self.vna = NanoVNAClient(config)
            self.vna.connect()
            self.msg_queue.put(("status", "Status: Connected. Recognition is running."))

            while not self.stop_event.is_set():
                re, im = self.vna.read_s11()
                X = preprocess_re_im(re, im, expected_points=self.expected_points)
                xb = torch.from_numpy(X).to(DEVICE)

                with torch.no_grad():
                    logits = self.model(xb)
                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

                order = np.argsort(probs)[::-1]
                top = [(self.label_names[int(i)], float(probs[int(i)])) for i in order[:3]]
                self.msg_queue.put(("prediction", top))
                time.sleep(max(0.05, config.interval_s))

        except Exception as e:
            self.msg_queue.put(("error", f"Runtime error: {e}"))
            self.msg_queue.put(("debug", traceback.format_exc()))
        finally:
            if self.vna is not None:
                self.vna.close()
            self.msg_queue.put(("stopped", "Status: Stopped"))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "status":
                    self.status_label.configure(text=payload)
                elif kind == "prediction":
                    self._update_prediction(payload)
                elif kind == "error":
                    self.status_label.configure(text=payload)
                    messagebox.showerror("Recognition Error", payload)
                elif kind == "info":
                    messagebox.showinfo("VNA Test", payload)
                elif kind == "debug":
                    print(payload)
                elif kind == "stopped":
                    self.status_label.configure(text=payload)
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _stable_decision(self, pred: str, conf: float) -> Tuple[Optional[str], Optional[float], str]:
        """Temporal voting + confidence gate.

        A single VNA frame can be unstable while the tag is moving toward or away
        from the reader. This function only updates the displayed result when the
        same class appears consistently in recent high-confidence frames.
        """
        if conf < CONFIDENCE_THRESHOLD:
            if KEEP_LAST_STABLE and self.stable_prediction is not None:
                return self.stable_prediction, self.stable_confidence, "Holding last stable result"
            return None, None, "Analyzing: low confidence"

        self.prediction_buffer.append(pred)
        self.confidence_buffer.append(conf)

        if len(self.prediction_buffer) < VOTE_WINDOW:
            if KEEP_LAST_STABLE and self.stable_prediction is not None:
                return self.stable_prediction, self.stable_confidence, "Collecting frames"
            return None, None, f"Collecting frames ({len(self.prediction_buffer)}/{VOTE_WINDOW})"

        counts = Counter(self.prediction_buffer)
        top_label, top_count = counts.most_common(1)[0]

        if top_count >= VOTE_MIN_COUNT:
            selected_conf = [
                c for p, c in zip(self.prediction_buffer, self.confidence_buffer)
                if p == top_label
            ]
            self.stable_prediction = top_label
            self.stable_confidence = float(np.mean(selected_conf))
            return self.stable_prediction, self.stable_confidence, "Stable"

        if KEEP_LAST_STABLE and self.stable_prediction is not None:
            return self.stable_prediction, self.stable_confidence, "Transition detected"
        return None, None, "Analyzing: unstable vote"

    def _update_prediction(self, top: List[Tuple[str, float]]):
        if not top:
            return

        live_pred, live_conf = top[0]
        stable_pred, stable_conf, status = self._stable_decision(live_pred, live_conf)

        if stable_pred is None:
            self.prediction_label.configure(text="Analyzing")
            self.confidence_label.configure(text="Confidence: --")
        else:
            self.prediction_label.configure(text=stable_pred)
            self.confidence_label.configure(text=f"Confidence: {stable_conf * 100:.1f}%")

        self.decision_status_label.configure(
            text=f"Decision Status: {status} | Live: {live_pred} ({live_conf * 100:.1f}%)"
        )

        labels = [self.top1_label, self.top2_label, self.top3_label]
        for i, lab in enumerate(labels):
            if i < len(top):
                name, p = top[i]
                lab.configure(text=f"#{i + 1}  {name}  ({p * 100:.1f}%)")
            else:
                lab.configure(text=f"#{i + 1}  --")

if __name__ == "__main__":
    root = tk.Tk()
    app = EMFingerprintApp(root)
    root.mainloop()
