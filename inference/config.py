"""
Centralized configuration for Vision-Based Fault Detection.

Usage:
    Set DEVICE_PROFILE=rasp or DEVICE_PROFILE=pc in your .env file
    to automatically load the correct model paths and tuning parameters.

    from inference.config import cfg
    print(cfg.DEVICE_PROFILE)       # "rasp" or "pc"
    print(cfg.CNN_MODEL_PATH)       # auto-resolved model path
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# Path Resolution
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


@dataclass
class DeviceConfig:
    """Configuration container for a specific device profile."""

    # ── Identity ──
    DEVICE_PROFILE: str = "pc"

    # ── Model Paths (resolved after init) ──
    YOLO_MODEL_PATH: str = ""
    CNN_MODEL_PATH: str = ""

    # ── Inference Backend ──
    USE_YOLO: bool = True          # Whether to use YOLO detector
    USE_CNN_REFINE: bool = True    # Whether to refine YOLO crops with CNN
    CNN_BACKEND: str = "pytorch"   # "pytorch" or "onnx"
    DEVICE: str = "cpu"            # "cpu" or "cuda"

    # ── Camera / Video Source ──
    IP_CAMERA_URL: str = "0"

    # ── Backend / API ──
    OCELOT_GATEWAY_URL: str = "http://localhost:5000"
    JWT_TOKEN: str = ""

    # ── Metadata ──
    TOWER_ID: str = "T-110KV-01"
    LATITUDE: float = 21.0285
    LONGITUDE: float = 105.8542

    # ── Tuning (Server/PC) ──
    PERSIST_FRAMES: int = 5
    HOLD_FRAMES: int = 15
    CLEANUP_FRAMES: int = 40
    MIN_REPORT_CONF: float = 0.40
    HISTORY_LEN: int = 15
    EMA_ALPHA: float = 0.3

    # ── Tuning (Gateway/Rasp) ──
    REPORT_COOLDOWN: float = 10.0  # Seconds between duplicate reports

    # ── Server ──
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000


# =====================================================================
# Profile Defaults
# =====================================================================
# These define the *defaults* for each profile. Any value can still be
# overridden by the corresponding environment variable.
# =====================================================================

_PROFILE_DEFAULTS = {
    "pc": {
        "USE_YOLO": True,
        "USE_CNN_REFINE": True,
        "CNN_BACKEND": "pytorch",
        "DEVICE": "cpu",            # Set to "cuda" if GPU available
        # Model paths (relative to models/)
        "_YOLO_FILENAME": "best_detector.pt",
        "_CNN_FILENAME": "best_model.pth",
        "_CNN_FALLBACK": "insulator_cnn.pth",
        "_MODELS_SUBDIR": "pc",     # Look in models/pc/ first
        # Tuning
        "PERSIST_FRAMES": 5,
        "HOLD_FRAMES": 15,
        "CLEANUP_FRAMES": 40,
        "MIN_REPORT_CONF": 0.40,
        "HISTORY_LEN": 15,
        "EMA_ALPHA": 0.3,
        "SERVER_PORT": 8000,
    },
    "rasp": {
        "USE_YOLO": False,          # Rasp typically runs CNN-only (lighter)
        "USE_CNN_REFINE": True,
        "CNN_BACKEND": "onnx",      # Use ONNX on edge device
        "DEVICE": "cpu",
        # Model paths
        "_YOLO_FILENAME": "",
        "_CNN_FILENAME": "best_model.onnx",
        "_CNN_FALLBACK": "",
        "_MODELS_SUBDIR": "rasp",   # Look in models/rasp/ first
        # Tuning (lighter for edge)
        "PERSIST_FRAMES": 3,
        "HOLD_FRAMES": 10,
        "CLEANUP_FRAMES": 30,
        "MIN_REPORT_CONF": 0.35,
        "HISTORY_LEN": 10,
        "EMA_ALPHA": 0.4,
        "REPORT_COOLDOWN": 10.0,
        "SERVER_PORT": 8000,
    },
}


def _resolve_model_path(subdir: str, filename: str, fallback: str = "") -> str:
    """
    Resolve model path with priority:
      1. models/<subdir>/<filename>   (profile-specific)
      2. models/<filename>            (shared root)
      3. models/<subdir>/<fallback>   (profile-specific fallback)
      4. models/<fallback>            (shared root fallback)
      5. ""                           (not found)
    """
    if not filename:
        return ""

    candidates = [
        os.path.join(MODELS_DIR, subdir, filename),
        os.path.join(MODELS_DIR, filename),
    ]
    if fallback:
        candidates.extend([
            os.path.join(MODELS_DIR, subdir, fallback),
            os.path.join(MODELS_DIR, fallback),
        ])

    for path in candidates:
        if os.path.exists(path):
            return path

    # Return the first candidate (for a helpful error message later)
    return candidates[0]


def _load_config() -> DeviceConfig:
    """Build a DeviceConfig by layering: base defaults → profile defaults → env vars."""

    profile = os.getenv("DEVICE_PROFILE", "pc").lower()
    if profile not in _PROFILE_DEFAULTS:
        print(f"⚠️  Unknown DEVICE_PROFILE='{profile}', falling back to 'pc'")
        profile = "pc"

    defaults = _PROFILE_DEFAULTS[profile]
    cfg = DeviceConfig(DEVICE_PROFILE=profile)

    # ── Apply profile defaults ──
    for key, value in defaults.items():
        if key.startswith("_"):
            continue  # Skip internal keys
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    # ── Override with environment variables ──
    env_mapping = {
        "IP_CAMERA_URL": str,
        "OCELOT_GATEWAY_URL": str,
        "JWT_TOKEN": str,
        "TOWER_ID": str,
        "LATITUDE": float,
        "LONGITUDE": float,
        "DEVICE": str,
        "CNN_BACKEND": str,
        "USE_YOLO": lambda v: v.lower() in ("true", "1", "yes"),
        "USE_CNN_REFINE": lambda v: v.lower() in ("true", "1", "yes"),
        "PERSIST_FRAMES": int,
        "HOLD_FRAMES": int,
        "CLEANUP_FRAMES": int,
        "MIN_REPORT_CONF": float,
        "HISTORY_LEN": int,
        "EMA_ALPHA": float,
        "REPORT_COOLDOWN": float,
        "SERVER_HOST": str,
        "SERVER_PORT": int,
    }

    for env_key, converter in env_mapping.items():
        env_val = os.getenv(env_key)
        if env_val is not None:
            try:
                setattr(cfg, env_key, converter(env_val))
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid env var {env_key}={env_val}: {e}")

    # ── Resolve model paths ──
    subdir = defaults.get("_MODELS_SUBDIR", "")

    # YOLO: env > profile default
    yolo_env = os.getenv("YOLO_MODEL_PATH")
    if yolo_env:
        cfg.YOLO_MODEL_PATH = yolo_env
    else:
        cfg.YOLO_MODEL_PATH = _resolve_model_path(
            subdir,
            defaults.get("_YOLO_FILENAME", ""),
        )

    # CNN: env > profile default
    cnn_env = os.getenv("CNN_MODEL_PATH") or os.getenv("ONNX_MODEL_PATH")
    if cnn_env:
        cfg.CNN_MODEL_PATH = cnn_env
    else:
        cfg.CNN_MODEL_PATH = _resolve_model_path(
            subdir,
            defaults.get("_CNN_FILENAME", ""),
            defaults.get("_CNN_FALLBACK", ""),
        )

    return cfg


# =====================================================================
# Singleton — import this everywhere
# =====================================================================
cfg = _load_config()


def print_config():
    """Pretty-print the active configuration."""
    print("=" * 65)
    print(f"🔧  Active Config — Profile: {cfg.DEVICE_PROFILE.upper()}")
    print("=" * 65)
    print(f"  YOLO Detector:   {'ON' if cfg.USE_YOLO else 'OFF'}")
    print(f"  YOLO Model:      {cfg.YOLO_MODEL_PATH or '(none)'}")
    print(f"  CNN Refiner:     {'ON' if cfg.USE_CNN_REFINE else 'OFF'}")
    print(f"  CNN Model:       {cfg.CNN_MODEL_PATH or '(none)'}")
    print(f"  CNN Backend:     {cfg.CNN_BACKEND}")
    print(f"  Device:          {cfg.DEVICE}")
    print(f"  Camera Source:   {cfg.IP_CAMERA_URL}")
    print(f"  API Gateway:     {cfg.OCELOT_GATEWAY_URL}")
    print(f"  Tower ID:        {cfg.TOWER_ID}")
    print(f"  Server:          {cfg.SERVER_HOST}:{cfg.SERVER_PORT}")
    print("=" * 65)
