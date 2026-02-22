import argparse
import ctypes
import errno
import gc
import traceback
import inspect
import importlib.util
import math
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import cv2
import gradio as gr
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor, Sam3Model, Sam3Processor

# -------------------------
# Job file I/O
# -------------------------

# job-level defaults (set after reading request.json)
_JOB_PROMPT_MODE = "click"
_JOB_OUTPUT_MODE = "fgmask"
_JOB_BG_MODE_DEFAULT = "GB"
_JOB_INSERT_MODE_DEFAULT = "transparent"
_JOB_GBBB_CODEC = "mp4v"
_JOB_AUDIO_MUX_DEFAULT = True
_JOB_DEVICE = "cuda"
_JOB_DTYPE = torch.bfloat16
_env_compile = str(os.environ.get("SAM3_TORCH_COMPILE", "auto")).strip().lower()
if _env_compile in ("1", "true", "yes", "on"):
    _JOB_USE_TORCH_COMPILE = True
elif _env_compile in ("0", "false", "no", "off"):
    _JOB_USE_TORCH_COMPILE = False
else:
    # torch.compile is unstable on some Python 3.13 builds.
    _JOB_USE_TORCH_COMPILE = sys.version_info < (3, 13)
_JOB_SESSION_PROCESSING_DEVICE = "cuda"
_JOB_SESSION_VIDEO_STORAGE_DEVICE = "cuda"
_JOB_SESSION_INFERENCE_STATE_DEVICE = "cuda"
_JOB_SESSION_CHUNK_FRAMES = 240
_JOB_PROPAGATE_CHUNK_MAX_FRAMES = 40
_JOB_FRAME_MEM_BUDGET_RATIO = 0.20
_JOB_FRAME_MAX_COUNT = 0
_JOB_ONDEMAND_CACHE_MAX_FRAMES = 24
_JOB_UI_WAIT_TIMEOUT_SEC = 7200
_COMPOSITED_CACHE_MAX_FRAMES = 48
_STATUS_WRITE_CONSECUTIVE_FAILS = 0
_STATUS_WRITE_LAST_ERROR = ""
_STATUS_WRITE_FATAL_THRESHOLD = 3

def _select_device(pref: str) -> str:
    p = str(pref or "").strip().lower()
    if p.startswith("cpu"):
        return "cpu"
    if p.startswith("cuda") or p in ("gpu", "auto", ""):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def _normalize_torch_device_name(value: str, fallback: str = "cpu") -> str:
    s = str(value or "").strip().lower()
    if s.startswith("cuda"):
        return "cuda"
    if s.startswith("cpu"):
        return "cpu"
    return fallback

def _default_session_devices(job_device: str) -> Tuple[str, str, str]:
    d = _normalize_torch_device_name(job_device, "cpu")
    if d == "cuda":
        # Prefer VRAM usage.
        return ("cuda", "cuda", "cuda")
    return ("cpu", "cpu", "cpu")

def _set_session_devices(processing_device: str, video_storage_device: str, inference_state_device: str) -> None:
    global _JOB_SESSION_PROCESSING_DEVICE, _JOB_SESSION_VIDEO_STORAGE_DEVICE, _JOB_SESSION_INFERENCE_STATE_DEVICE
    _JOB_SESSION_PROCESSING_DEVICE = _normalize_torch_device_name(processing_device, "cpu")
    _JOB_SESSION_VIDEO_STORAGE_DEVICE = _normalize_torch_device_name(video_storage_device, "cpu")
    _JOB_SESSION_INFERENCE_STATE_DEVICE = _normalize_torch_device_name(inference_state_device, "cpu")

def _session_device_profiles() -> List[Tuple[str, str, str]]:
    base = (
        _JOB_SESSION_PROCESSING_DEVICE,
        _JOB_SESSION_VIDEO_STORAGE_DEVICE,
        _JOB_SESSION_INFERENCE_STATE_DEVICE,
    )
    profiles: List[Tuple[str, str, str]] = [base]
    if _normalize_torch_device_name(_JOB_DEVICE, "cpu") == "cuda":
        for cand in (
            ("cuda", "cpu", "cuda"),
            ("cuda", "cpu", "cpu"),
            ("cpu", "cpu", "cpu"),
        ):
            if cand not in profiles:
                profiles.append(cand)
    elif ("cpu", "cpu", "cpu") not in profiles:
        profiles.append(("cpu", "cpu", "cpu"))
    return profiles

def _select_dtype(device: str, pref: str = "bf16") -> torch.dtype:
    d = (device or "cpu").lower()
    p = (pref or "bf16").lower()
    if d == "cpu":
        return torch.float32
    if d in ("cuda", "mps"):
        if d == "cuda" and p in ("bf16", "bfloat16"):
            try:
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
        # fallback
        return torch.float16
    return torch.float32

def _apply_request_defaults(req: dict) -> None:
    """Stage1: apply request.json job-level defaults."""
    global _JOB_PROMPT_MODE, _JOB_OUTPUT_MODE, _JOB_BG_MODE_DEFAULT, _JOB_GBBB_CODEC, _JOB_AUDIO_MUX_DEFAULT
    global _JOB_DEVICE, _JOB_DTYPE, _JOB_INSERT_MODE_DEFAULT
    global _JOB_SESSION_CHUNK_FRAMES, _JOB_PROPAGATE_CHUNK_MAX_FRAMES, _JOB_FRAME_MEM_BUDGET_RATIO, _JOB_FRAME_MAX_COUNT, _JOB_ONDEMAND_CACHE_MAX_FRAMES, _JOB_UI_WAIT_TIMEOUT_SEC

    _JOB_PROMPT_MODE = str((req.get("prompt", {}) or {}).get("mode", "click") or "click")
    _JOB_OUTPUT_MODE = str((req.get("output", {}) or {}).get("mode", "fgmask") or "fgmask")
    out = req.get("output", {}) or {}
    _JOB_BG_MODE_DEFAULT = str(out.get("bg_mode", "GB") or "GB").upper()
    _JOB_INSERT_MODE_DEFAULT = str(out.get("insert_mode", "transparent") or "transparent")

    gbbb = out.get("gbbb", {}) or {}
    _JOB_GBBB_CODEC = str(gbbb.get("codec", "mp4v") or "mp4v")
    _JOB_AUDIO_MUX_DEFAULT = bool(gbbb.get("audio_mux", True))
    opts = req.get("options", {}) or {}
    requested_device = str(opts.get("device_preference", "cuda") or "cuda")
    _JOB_DEVICE = _select_device(requested_device)
    if _JOB_DEVICE == "cpu" and requested_device.strip().lower().startswith("cuda"):
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
    _JOB_DTYPE = _select_dtype(_JOB_DEVICE, str(opts.get("dtype_preference", "bf16") or "bf16"))
    def_p, def_v, def_s = _default_session_devices(_JOB_DEVICE)
    sess = opts.get("session_devices", {}) or {}
    _set_session_devices(
        sess.get("processing_device", def_p),
        sess.get("video_storage_device", def_v),
        sess.get("inference_state_device", def_s),
    )
    if _JOB_DEVICE != "cuda":
        _set_session_devices("cpu", "cpu", "cpu")
    raw_chunk = opts.get("session_chunk_frames", 240)
    try:
        chunk = int(raw_chunk)
    except Exception:
        chunk = 240
    _JOB_SESSION_CHUNK_FRAMES = min(4096, max(32, chunk))

    raw_prop_chunk = opts.get("propagate_chunk_max_frames", opts.get("propagate_chunk_frames", 40))
    try:
        prop_chunk = int(raw_prop_chunk)
    except Exception:
        prop_chunk = 40
    _JOB_PROPAGATE_CHUNK_MAX_FRAMES = min(512, max(32, prop_chunk))

    # Frame load memory policy:
    # - ratio: portion of currently available system memory used for decoded frame storage
    # - max_count: hard cap for loaded frame count (0 = disabled)
    raw_ratio = opts.get("frame_mem_budget_ratio", opts.get("video_frame_mem_budget_ratio", 0.20))
    try:
        ratio = float(raw_ratio)
    except Exception:
        ratio = 0.20
    _JOB_FRAME_MEM_BUDGET_RATIO = min(0.60, max(0.05, ratio))

    raw_cap = opts.get("frame_max_count", opts.get("video_frame_max_count", 0))
    try:
        cap = int(raw_cap)
    except Exception:
        cap = 0
    _JOB_FRAME_MAX_COUNT = max(0, cap)

    raw_od_cache = opts.get("ondemand_cache_max_frames", 24)
    try:
        od_cache = int(raw_od_cache)
    except Exception:
        od_cache = 24
    _JOB_ONDEMAND_CACHE_MAX_FRAMES = min(512, max(8, od_cache))

    raw_ui_wait = opts.get("ui_wait_timeout_sec", 7200)
    try:
        ui_wait = int(raw_ui_wait)
    except Exception:
        ui_wait = 7200
    if ui_wait < 0:
        ui_wait = 0
    _JOB_UI_WAIT_TIMEOUT_SEC = min(86400, ui_wait)

def _ensure_cuda_triton_ready() -> None:
    global _JOB_DEVICE, _JOB_DTYPE, _JOB_USE_TORCH_COMPILE
    if _JOB_DEVICE != "cuda":
        _JOB_USE_TORCH_COMPILE = False
        return
    if not torch.cuda.is_available():
        print("[WARN] CUDA device is unavailable at runtime. Falling back to CPU.")
        _JOB_DEVICE = "cpu"
        _JOB_DTYPE = torch.float32
        _set_session_devices("cpu", "cpu", "cpu")
        _JOB_USE_TORCH_COMPILE = False
        return
    if importlib.util.find_spec("triton") is None:
        if _JOB_USE_TORCH_COMPILE:
            print("[WARN] Triton was not found. torch.compile is disabled.")
        _JOB_USE_TORCH_COMPILE = False

def _ensure_pyav_ready() -> None:
    try:
        import av  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "PyAV (package `av`) is required for SAM3 video sessions. "
            "Install it with `pip install av` or reinstall `python/requirements-cuda.txt`."
        ) from e

def _configure_cuda_runtime() -> None:
    # Fast-path defaults for NVIDIA GPU inference.
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def _available_system_memory_bytes() -> int:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        mem = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
            return int(mem.ullAvailPhys)
        return 0

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except Exception:
        return 0

def jst_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _is_retryable_replace_error(exc: BaseException) -> bool:
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError):
        winerror = int(getattr(exc, "winerror", 0) or 0)
        if winerror in (5, 32):
            return True
        err_no = int(getattr(exc, "errno", 0) or 0)
        if err_no in (errno.EACCES, errno.EPERM, errno.EBUSY):
            return True
    return False

def write_atomic(path: str, text: str, retries: int = 30, retry_sleep: float = 0.02) -> None:
    # Use unique temp files to avoid collisions when status polling is frequent.
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            pass

    last_exc: Optional[BaseException] = None
    for i in range(max(1, int(retries))):
        try:
            os.replace(tmp, path)
            return
        except Exception as e:
            last_exc = e
            if (not _is_retryable_replace_error(e)) or (i + 1 >= max(1, int(retries))):
                break
            time.sleep(float(retry_sleep))

    # Fallback: non-atomic write so job progression does not fail on transient lock.
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return
    except Exception:
        if last_exc is not None:
            raise last_exc
        raise
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def _safe_remove_file(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except Exception as e:
        print(f"[WARN] remove file failed: {path} ({e})")

def _norm_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(str(path or "")))

def _safe_replace_file(src_path: str, dst_path: str, retries: int = 30, retry_sleep: float = 0.02) -> str:
    src = str(src_path or "")
    dst = str(dst_path or "")
    if not src:
        raise RuntimeError("replace source path is empty")
    if not os.path.isfile(src):
        raise RuntimeError(f"replace source file not found: {src}")
    if not dst:
        raise RuntimeError("replace destination path is empty")

    if _norm_path(src) == _norm_path(dst):
        return dst

    dst_parent = os.path.dirname(dst)
    if dst_parent:
        os.makedirs(dst_parent, exist_ok=True)

    last_exc: Optional[BaseException] = None
    for i in range(max(1, int(retries))):
        try:
            if os.path.isfile(dst):
                _safe_remove_file(dst)
            os.replace(src, dst)
            return dst
        except Exception as e:
            last_exc = e
            if (not _is_retryable_replace_error(e)) or (i + 1 >= max(1, int(retries))):
                break
            time.sleep(float(retry_sleep))

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"replace failed: {src} -> {dst}")

def _sanitize_output_stem(name: str) -> str:
    stem = str(name or "").strip()
    if not stem:
        return "sam3_output"
    table = str.maketrans({
        "<": "_", ">": "_", ":": "_", "\"": "_",
        "/": "_", "\\": "_", "|": "_", "?": "_", "*": "_",
    })
    stem = stem.translate(table).strip().rstrip(". ")
    return stem or "sam3_output"

def _source_stem_for_output(source_video_path: str) -> str:
    stem = Path(str(source_video_path or "")).stem
    return _sanitize_output_stem(stem)

def _cleanup_job_output_artifacts(output_dir: str, job_id: str, keep_paths: Sequence[str]) -> None:
    out_dir = Path(str(output_dir or ""))
    if not out_dir.is_dir():
        return
    token = f"_{str(job_id or '').strip()}_"
    if not token or token == "__":
        return
    keep = {_norm_path(p) for p in keep_paths if p}
    for p in out_dir.iterdir():
        try:
            if (not p.is_file()) or (token not in p.name):
                continue
            if _norm_path(str(p)) in keep:
                continue
            _safe_remove_file(str(p))
        except Exception as e:
            print(f"[WARN] cleanup artifact failed: {p} ({e})")

def write_status(job_dir: str, **kwargs) -> None:
    global _STATUS_WRITE_CONSECUTIVE_FAILS, _STATUS_WRITE_LAST_ERROR
    payload = {
        "state": kwargs.get("state", "running"),
        "prompt_mode": kwargs.get("prompt_mode", _JOB_PROMPT_MODE),
        "output_mode": kwargs.get("output_mode", _JOB_OUTPUT_MODE),
        "phase": kwargs.get("phase", ""),
        "progress": float(kwargs.get("progress", 0.0)),
        "message": kwargs.get("message", ""),
        "gradio_url": kwargs.get("gradio_url", ""),
        "updated_at_jst": jst_now(),
    }
    try:
        write_atomic(os.path.join(job_dir, "status.json"), json.dumps(payload, ensure_ascii=False, indent=2))
        _STATUS_WRITE_CONSECUTIVE_FAILS = 0
        _STATUS_WRITE_LAST_ERROR = ""
    except Exception as e:
        _STATUS_WRITE_CONSECUTIVE_FAILS += 1
        _STATUS_WRITE_LAST_ERROR = str(e)
        print(f"[WARN] write_status failed ({_STATUS_WRITE_CONSECUTIVE_FAILS}): {e}")
        if _STATUS_WRITE_CONSECUTIVE_FAILS >= _STATUS_WRITE_FATAL_THRESHOLD:
            raise RuntimeError(
                f"status.json write failed repeatedly: {_STATUS_WRITE_LAST_ERROR}"
            ) from e

def write_result(
    job_dir: str,
    success: bool,
    mask_video_path: str = "",
    output_video_path: str = "",
    fg_video_path: str = "",
    composited_video_path: str = "",
    bg_mode: str = "",
    insert_mode: str = "",
    error_message: str = "",
    stats: Optional[dict] = None,
    prompt_mode: Optional[str] = None,
    output_mode: Optional[str] = None,
) -> None:
    pm = _JOB_PROMPT_MODE if prompt_mode is None else prompt_mode
    om = _JOB_OUTPUT_MODE if output_mode is None else output_mode
    st_in = stats if isinstance(stats, Mapping) else {}
    st = dict(st_in)
    try:
        num_frames = int(st.get("num_frames", 0) or 0)
    except Exception:
        num_frames = 0
    try:
        fps = float(st.get("fps", 0.0) or 0.0)
    except Exception:
        fps = 0.0
    st["num_frames"] = num_frames
    st["fps"] = fps

    payload = {
        "success": bool(success),
        "prompt_mode": pm,
        "output_mode": om,
        "mask_video_path": mask_video_path,
        "output_video_path": output_video_path,
        "fg_video_path": fg_video_path,
        "composited_video_path": composited_video_path,
        "bg_mode": bg_mode,
        "insert_mode": insert_mode,
        "error_message": error_message,
        "stats": st,
        # Keep top-level fields for launcher compatibility.
        "num_frames": num_frames,
        "fps": fps,
    }
    write_atomic(os.path.join(job_dir, "result.json"), json.dumps(payload, ensure_ascii=False, indent=2))

# -------------------------
# Environment / logging
# -------------------------

def setup_logging(job_dir: Path) -> None:
    log_fp = job_dir / "python.log.txt"
    log_fp.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_fp, "a", encoding="utf-8", buffering=1)
    sys.stdout = log_f
    sys.stderr = log_f

def setup_proxy_safety() -> None:
    # Ensure loopback addresses bypass proxy while preserving existing settings.
    required = ["127.0.0.1", "localhost"]
    for key in ("NO_PROXY", "no_proxy"):
        cur = os.environ.get(key, "")
        parts = [p.strip() for p in cur.split(",") if p.strip()]
        for v in required:
            if v not in parts:
                parts.append(v)
        os.environ[key] = ",".join(parts)
# -------------------------
# Video I/O (segment load)
# -------------------------

@dataclass
class SegmentInfo:
    fps: float
    num_frames: int
    width: int
    height: int
    start_sec: float
    end_sec: float

class OnDemandVideoFrames:
    def __init__(
        self,
        video_path: str,
        num_frames: int,
        fps: float,
        width: int,
        height: int,
        cache_max: int = 24,
    ) -> None:
        self.video_path = str(video_path)
        self.num_frames = max(1, int(num_frames))
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.width = int(width)
        self.height = int(height)
        self.cache_max = max(8, int(cache_max))

        self._lock = threading.Lock()
        self._cap = None
        self._last_idx = -1
        self._source_num_frames = 0
        self._cache: "OrderedDict[int, Image.Image]" = OrderedDict()

        cap = cv2.VideoCapture(self.video_path)
        if cap is not None and cap.isOpened():
            try:
                self._source_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            except Exception:
                self._source_num_frames = 0
            try:
                fps_m = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                if fps_m > 0:
                    self.fps = fps_m
            except Exception:
                pass
            try:
                w_m = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h_m = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if w_m > 0 and h_m > 0:
                    self.width = w_m
                    self.height = h_m
            except Exception:
                pass
            cap.release()

    def __len__(self) -> int:
        return int(self.num_frames)

    def __bool__(self) -> bool:
        return self.num_frames > 0

    def __getitem__(self, idx: int) -> Image.Image:
        return self.get(int(idx))

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._last_idx = -1
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
            self._cap = None

    def _ensure_cap_locked(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return
        self._cap = cv2.VideoCapture(self.video_path)
        if self._cap is None or (not self._cap.isOpened()):
            raise RuntimeError(f"Failed to open segment video: {self.video_path}")
        self._last_idx = -1

    def _decode_frame_locked(self, decode_idx: int) -> Image.Image:
        self._ensure_cap_locked()
        ret = False
        frame_bgr = None

        if self._last_idx >= 0 and decode_idx == (self._last_idx + 1):
            ret, frame_bgr = self._cap.read()
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(decode_idx))
            ret, frame_bgr = self._cap.read()
            if (not ret) and self.fps > 0:
                self._cap.set(cv2.CAP_PROP_POS_MSEC, (float(decode_idx) / self.fps) * 1000.0)
                ret, frame_bgr = self._cap.read()

        if not ret or frame_bgr is None:
            raise RuntimeError(f"Failed to decode frame at index={decode_idx}")

        self._last_idx = int(decode_idx)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def get(self, idx: int) -> Image.Image:
        i = int(idx)
        if i < 0:
            i = 0
        if i >= self.num_frames:
            i = self.num_frames - 1

        with self._lock:
            cached = self._cache.get(i)
            if cached is not None:
                self._cache.move_to_end(i)
                return cached

            # Try the requested index first. Some codecs return unreliable
            # CAP_PROP_FRAME_COUNT values, so eager clamping can freeze all
            # frames to index 0/last frame.
            decode_candidates: List[int] = [int(i)]
            if self._source_num_frames > 0:
                meta_last = int(self._source_num_frames - 1)
                if meta_last >= 0 and meta_last != int(i):
                    decode_candidates.append(meta_last)

            img = None
            last_exc: Optional[Exception] = None
            for decode_idx in decode_candidates:
                if decode_idx < 0:
                    continue
                try:
                    img = self._decode_frame_locked(int(decode_idx))
                    break
                except Exception as e:
                    last_exc = e
                    continue
            if img is None:
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError(f"Failed to decode frame at index={i}")

            self._cache[i] = img
            while len(self._cache) > self.cache_max:
                self._cache.popitem(last=False)
            return img

def load_video_segment_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    max_seconds: int = 600,
    expected_num_frames: Optional[int] = None,
    load_all: bool = True,
) -> Tuple[List[Image.Image], SegmentInfo]:
    """
    髫ｰ謔ｶ繝ｻ繝ｻ・ｮ陞｢・ｼ驍・・・ｫ・｢郢晢ｽｻ[start_sec, end_sec] 驛｢・ｧ陞ｳ螟ｲ・ｽ・ｪ繝ｻ・ｭ驍ｵ・ｺ繝ｻ・ｿ鬮ｴ雜｣・ｽ・ｼ驍ｵ・ｺ繝ｻ・ｿ驍ｵ・ｲ郢晢ｽｻ
    end_sec <= start_sec 驍ｵ・ｺ繝ｻ・ｮ驍ｵ・ｺ繝ｻ・ｨ驍ｵ・ｺ鬮ｦ・ｪ郢晢ｽｻ start 驍ｵ・ｺ闕ｵ譎｢・ｽ繝ｻmax_seconds 髯具ｽｻ郢晢ｽｻ郢晢ｽｻ
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Invalid ranges should fail fast. The plugin always provides an explicit segment.
    if end_sec <= start_sec:
        cap.release()
        raise RuntimeError(f"Invalid segment range: start_sec={start_sec}, end_sec={end_sec}")

    # max_seconds 髯橸ｽｳ霑壼生繝ｻ髯滉ｻ｣繝ｻ
    if max_seconds and (end_sec - start_sec) > float(max_seconds):
        end_sec = start_sec + float(max_seconds)

    # 髯ｷ・ｿ繝ｻ・ｯ鬮｢・ｭ繝ｻ・ｽ驍ｵ・ｺ繝ｻ・ｪ驛｢・ｧ騾包ｽｻ陷・ｽｾ髯具ｽｻ繝ｻ・ｻ驛｢・ｧ繝ｻ・ｷ驛｢譎｢・ｽ・ｼ驛｢・ｧ繝ｻ・ｯ
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_sec) * 1000.0)

    frames: List[Image.Image] = []
    t_limit = end_sec
    # cv2 驍ｵ・ｺ繝ｻ・ｯ timestamp 髯ｷ・ｿ鬮｢ﾂ繝ｻ・ｾ陷会ｽｱ遯ｶ・ｲ髣包ｽｳ隶主･・ｽｽ・ｮ霑壼遜・ｽ・ｮ陞｢・ｹ遶企・・ｸ・ｺ髦ｮ蜷ｮ繝ｻ驍ｵ・ｺ陟募ｨｯ譌ｺ驛｢・ｧ闕ｵ譏ｴ繝ｻ驍ｵ・ｺ繝ｻ・ｧ驍ｵ・ｲ遯ｶ蜀ｪs髫ｰ・ｰ陝ｶ・ｷ繝ｻ・ｮ陷会ｽｱ繝ｻ繧頑割繝ｻ・ｵ鬨ｾ蛹・ｽｽ・ｨ
    start_frame_guess = int(round(start_sec * fps))
    exp_frames = int(expected_num_frames) if expected_num_frames is not None else 0
    if exp_frames > 0:
        end_frame_guess = start_frame_guess + exp_frames
    else:
        end_frame_guess = int(round(end_sec * fps))

    # Keep frame storage within a safe portion of free host memory.
    # Use a conservative bytes-per-frame estimate because PIL + numpy conversion
    # and downstream mask buffers increase real usage beyond raw RGB size.
    req_frames = max(1, end_frame_guess - start_frame_guess)
    if not load_all:
        hard_cap = 0
        if _JOB_FRAME_MAX_COUNT > 0:
            hard_cap = max(16, int(_JOB_FRAME_MAX_COUNT))
        if hard_cap > 0 and req_frames > hard_cap:
            req_frames = hard_cap

        available_from_start = 0
        if source_total_frames > 0:
            available_from_start = max(0, int(source_total_frames) - max(0, int(start_frame_guess)))

        # Prefer actual decodable frame count in the preprocessed segment.
        target_num_frames = req_frames
        if available_from_start > 0:
            target_num_frames = max(1, int(available_from_start))
        if exp_frames > 0:
            if available_from_start > 0:
                target_num_frames = min(target_num_frames, int(exp_frames))
            else:
                target_num_frames = int(exp_frames)
        if hard_cap > 0:
            target_num_frames = min(target_num_frames, hard_cap)
        target_num_frames = max(1, int(target_num_frames))

        first_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        if first_frame_index < start_frame_guess - 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_guess)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame_guess))
            ret, frame_bgr = cap.read()
        cap.release()
        if not ret or frame_bgr is None:
            raise RuntimeError("No frames loaded in the requested segment.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        first_frame = Image.fromarray(frame_rgb)
        if w <= 0 or h <= 0:
            w, h = first_frame.size

        seg_end = float(start_sec + (target_num_frames / max(1e-6, fps)))
        info = SegmentInfo(
            fps=fps,
            num_frames=target_num_frames,
            width=w,
            height=h,
            start_sec=float(start_sec),
            end_sec=seg_end,
        )
        return [first_frame], info

    frame_cap = 0
    if w > 0 and h > 0 and fps > 0:
        avail_mem = _available_system_memory_bytes()
        if avail_mem > 0:
            mem_budget = int(avail_mem * float(_JOB_FRAME_MEM_BUDGET_RATIO))
            bytes_per_frame = int(w) * int(h) * 4
            if bytes_per_frame > 0:
                frame_cap = max(16, mem_budget // bytes_per_frame)
    if _JOB_FRAME_MAX_COUNT > 0:
        hard_cap = max(16, int(_JOB_FRAME_MAX_COUNT))
        frame_cap = hard_cap if frame_cap <= 0 else min(frame_cap, hard_cap)

    if frame_cap > 0 and req_frames > frame_cap:
        end_frame_guess = start_frame_guess + frame_cap
        end_sec = start_sec + (frame_cap / fps)
        t_limit = end_sec
        if exp_frames > 0:
            exp_frames = frame_cap
        print(
            "[WARN] Frame load was capped by memory policy:",
            f"requested={req_frames}, cap={frame_cap}, ratio={_JOB_FRAME_MEM_BUDGET_RATIO:.3f},",
            f"hard_cap={_JOB_FRAME_MAX_COUNT}",
        )

    # Current frame position.
    cur_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    if cur_idx < start_frame_guess - 2:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_guess)

    while True:
        if exp_frames > 0 and len(frames) >= exp_frames:
            break
        cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        if exp_frames <= 0 and cur_frame >= end_frame_guess:
            break

        ret, frame_bgr = cap.read()
        if not ret:
            break

        # timestamp check (if available)
        if exp_frames <= 0:
            t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if t_msec and (t_msec / 1000.0) > t_limit:
                break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

        if len(frames) % 120 == 0:
            gc.collect()

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames loaded in the requested segment.")
    if exp_frames > 0 and len(frames) < exp_frames:
        short = exp_frames - len(frames)
        if short <= 2:
            print(f"[WARN] Segment was short by {short} frame(s). Padding with last frame.")
            last = frames[-1]
            for _ in range(short):
                frames.append(last.copy())
        else:
            raise RuntimeError(f"Segment frame shortage: expected={exp_frames}, got={len(frames)}")

    # Fallback to frame size when width/height cannot be read from metadata.
    if w <= 0 or h <= 0:
        w, h = frames[0].size

    info = SegmentInfo(
        fps=fps,
        num_frames=len(frames),
        width=w,
        height=h,
        start_sec=float(start_sec),
        end_sec=float(end_sec),
    )
    return frames, info

# -------------------------
# FFmpeg helpers (segment audio mux)
# -------------------------

def _resolve_ffmpeg_exe() -> str:
    # 1) imageio-ffmpeg bundled
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass

    # 2) env var
    env_exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if env_exe and os.path.isfile(env_exe):
        return env_exe

    # 3) system
    sys_exe = shutil.which("ffmpeg")
    if sys_exe:
        return sys_exe

    raise RuntimeError(
        "ffmpeg executable was not found. Install imageio-ffmpeg or set IMAGEIO_FFMPEG_EXE."
    )

def _decode_subprocess_text(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if not isinstance(data, (bytes, bytearray)):
        return str(data)
    b = bytes(data)
    encodings = ["utf-8", "cp932", sys.getdefaultencoding() or "utf-8"]
    seen = set()
    for enc in encodings:
        if enc in seen:
            continue
        seen.add(enc)
        try:
            return b.decode(enc)
        except Exception:
            pass
    return b.decode("utf-8", errors="replace")

def _run_ffmpeg(cmd: List[str]) -> Tuple[int, str]:
    # Read as bytes and decode ourselves to avoid locale-dependent UnicodeDecodeError.
    r = subprocess.run(cmd, capture_output=True, text=False)
    stderr = _decode_subprocess_text(r.stderr)
    if stderr:
        return r.returncode, stderr
    return r.returncode, _decode_subprocess_text(r.stdout)

def _probe_video_frame_count(path: str) -> Tuple[int, float]:
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if cap is None or not cap.isOpened():
            return 0, 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return n, fps
    except Exception:
        return 0, 0.0
    finally:
        if cap is not None:
            cap.release()

def preprocess_video_segment_for_tracking(
    source_video: str,
    out_path: str,
    start_sec: float,
    duration_sec: float,
    fps: Optional[float] = None,
    expected_num_frames: Optional[int] = None,
) -> str:
    if not source_video or not os.path.isfile(source_video):
        raise RuntimeError(f"source video not found: {source_video}")
    if duration_sec <= 0.0:
        raise RuntimeError(f"invalid segment duration: {duration_sec}")

    ffmpeg = _resolve_ffmpeg_exe()
    out_parent = os.path.dirname(out_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except Exception:
            pass

    vf = None
    fps_v = float(fps) if (fps is not None and float(fps) > 0.0) else 0.0
    exp_frames = int(expected_num_frames) if expected_num_frames is not None else 0
    if fps_v > 0.0:
        vf = f"fps={fps_v:.8f}"

    # Add a small headroom when frame count is explicit, then clamp by -frames:v.
    duration_for_cut = float(duration_sec)
    if fps_v > 0.0 and exp_frames > 0:
        duration_for_cut = max(duration_for_cut, (exp_frames / fps_v) + (0.5 / fps_v))

    def _build_cmd(codec_args: List[str], accurate_seek: bool, pad_tail: bool) -> List[str]:
        # accurate_seek=True keeps frame alignment more stable on fractional/inexact timestamps.
        if accurate_seek:
            cmd = [
                ffmpeg, "-y",
                "-i", source_video,
                "-ss", f"{start_sec:.6f}",
                "-t", f"{duration_for_cut:.6f}",
                "-map", "0:v:0",
                "-an", "-sn", "-dn",
            ]
        else:
            cmd = [
                ffmpeg, "-y",
                "-ss", f"{start_sec:.6f}",
                "-i", source_video,
                "-t", f"{duration_for_cut:.6f}",
                "-map", "0:v:0",
                "-an", "-sn", "-dn",
            ]

        vf_local = vf
        if pad_tail and fps_v > 0.0:
            # If the cut lands near EOF and comes short by 1 frame, clone last frame to fill.
            pad_sec = max(0.001, 2.0 / fps_v)
            tpad = f"tpad=stop_mode=clone:stop_duration={pad_sec:.8f}"
            vf_local = f"{vf_local},{tpad}" if vf_local else tpad

        if vf_local:
            cmd += ["-vf", vf_local]
        cmd += codec_args
        if exp_frames > 0:
            cmd += ["-frames:v", str(exp_frames), "-fps_mode", "cfr"]
        cmd += [out_path]
        return cmd

    def _output_ok(label: str) -> Tuple[bool, str]:
        if not os.path.isfile(out_path) or os.path.getsize(out_path) <= 0:
            return False, f"[{label}] output file missing or empty"
        if exp_frames <= 0:
            return True, ""

        meta_frames, meta_fps = _probe_video_frame_count(out_path)
        print(
            f"[FFMPEG_PREPROCESS] {label}: "
            f"meta_frames={meta_frames}, meta_fps={meta_fps:.6f}, expected_frames={exp_frames}"
        )
        if meta_frames > 0 and meta_frames < exp_frames:
            return False, f"[{label}] frame-shortage: got={meta_frames}, expected={exp_frames}"
        return True, ""

    codec_nvenc = [
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-cq", "23",
        "-b:v", "0",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    codec_x264 = [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]

    attempts: List[Tuple[str, List[str]]] = [
        ("nvenc-fast", _build_cmd(codec_nvenc, accurate_seek=False, pad_tail=False)),
        ("x264-fast", _build_cmd(codec_x264, accurate_seek=False, pad_tail=False)),
    ]
    if exp_frames > 0:
        attempts += [
            ("nvenc-accurate-pad", _build_cmd(codec_nvenc, accurate_seek=True, pad_tail=True)),
            ("x264-accurate-pad", _build_cmd(codec_x264, accurate_seek=True, pad_tail=True)),
        ]

    errors: List[str] = []
    for label, cmd in attempts:
        print(f"[FFMPEG_PREPROCESS] try={label}")
        print(f"[FFMPEG_PREPROCESS] cmd={subprocess.list2cmdline(cmd)}")
        code, err = _run_ffmpeg(cmd)
        if code != 0:
            errors.append(f"[{label}] exit={code}\n{err}")
            continue

        ok, msg = _output_ok(label)
        if ok:
            return out_path

        errors.append(f"{msg}\n{err}")
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass

    raise RuntimeError(
        "Failed to preprocess segment video.\n" + "\n\n".join(errors)
    )

def make_h264_aac_preview(
    video_noaudio: str,
    original_video: str,
    out_path: str,
    start_sec: float,
    end_sec: float,
    max_width: int = 960,
    crf: int = 23,
    preset: str = "veryfast",
    fps: Optional[int] = None,
    audio_bitrate: str = "96k",
) -> None:
    ffmpeg = _resolve_ffmpeg_exe()

    if not video_noaudio or not os.path.isfile(video_noaudio):
        raise RuntimeError(f"video_noaudio not found: {video_noaudio}")

    # scale: 髯晢ｽｷ郢晢ｽｻ繝ｻ遏･ax_width髣比ｼ夲ｽｽ・･髣包ｽｳ闕ｵ譏ｶ繝ｻ驍ｵ・ｺ陷会ｽｱ遯ｶ・ｻ鬩搾ｽｵ繝ｻ・ｦ驍ｵ・ｺ繝ｻ・ｯ鬮｢・ｾ繝ｻ・ｪ髯ｷ蟠趣ｽｼ譁舌・2驍ｵ・ｺ繝ｻ・ｧ髯句ｹ｢・ｽ・ｶ髫ｰ・ｨ繝ｻ・ｰ驍ｵ・ｺ繝ｻ・ｫ髫ｰ・ｰ郢晢ｽｻ遶擾ｽｴ驛｢・ｧ郢晢ｽｻ
    # min(960,iw) 驍ｵ・ｺ繝ｻ・ｮ 驕ｯ・ｶ郢晢ｽｻ驕ｯ・ｶ郢晢ｽｻ驍ｵ・ｺ繝ｻ・ｯ驛｢譎・ｽｼ譁絶襖驛｢譎｢・ｽ・ｫ驛｢・ｧ繝ｻ・ｿ髯ｷﾂ郢晢ｽｻ邵ｲ螳壼ｳｪ繝ｻ・ｺ髯具ｽｻ郢晢ｽｻ繝ｻ鬘假ｽｬ繝ｻ・ｽ・ｱ驍ｵ・ｺ郢晢ｽｻ遶企・・ｸ・ｺ繝ｻ・ｮ驍ｵ・ｺ繝ｻ・ｧ \, 驍ｵ・ｺ繝ｻ・ｧ驛｢・ｧ繝ｻ・ｨ驛｢・ｧ繝ｻ・ｹ驛｢・ｧ繝ｻ・ｱ驛｢譎｢・ｽ・ｼ驛｢譏ｴ繝ｻ
    vf = f"scale='min({max_width}\\,iw)':-2"
    if fps is not None and int(fps) > 0:
        vf = vf + f",fps={int(fps)}"

    cmd = [
        ffmpeg, "-y",
        "-i", video_noaudio,
        "-ss", f"{start_sec:.6f}",
        "-to", f"{end_sec:.6f}",
        "-i", original_video,

        "-map", "0:v:0",
        "-map", "1:a:0?",

        "-vf", vf,
        "-c:v", "libx264",
        "-preset", str(preset),
        "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        "-tag:v", "avc1",
        "-movflags", "+faststart",

        "-c:a", "aac",
        "-b:a", str(audio_bitrate),

        "-shortest",
        out_path,
    ]

    code, err = _run_ffmpeg(cmd)
    if code != 0:
        raise RuntimeError(f"Failed to make preview (H.264/AAC).\n{err}")

def make_h264_preview_video_only(
    video_noaudio: str,
    out_path: str,
    max_width: int = 960,
    crf: int = 23,
    preset: str = "veryfast",
    fps: Optional[int] = None,
) -> None:
    ffmpeg = _resolve_ffmpeg_exe()
    if not video_noaudio or not os.path.isfile(video_noaudio):
        raise RuntimeError(f"video_noaudio not found: {video_noaudio}")

    vf = f"scale='min({max_width}\\,iw)':-2"
    if fps is not None and int(fps) > 0:
        vf = vf + f",fps={int(fps)}"

    cmd = [
        ffmpeg, "-y",
        "-i", video_noaudio,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", str(preset),
        "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        "-tag:v", "avc1",
        "-movflags", "+faststart",
        "-an",
        out_path,
    ]
    code, err = _run_ffmpeg(cmd)
    if code != 0:
        raise RuntimeError(f"Failed to make preview (H.264 video only).\n{err}")

def mux_audio_from_original_segment(
    video_noaudio: str,
    original_video: str,
    out_path: str,
    start_sec: float,
    end_sec: float,
) -> None:
    ffmpeg = _resolve_ffmpeg_exe()

    # Prefer stream copy, fallback to AAC re-encode when copy is unavailable.
    cmd_copy = [
        ffmpeg, "-y",
        "-ss", f"{start_sec:.6f}",
        "-to", f"{end_sec:.6f}",
        "-i", original_video,
        "-i", video_noaudio,
        "-map", "1:v:0",
        "-map", "0:a:0?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        out_path,
    ]
    code, err = _run_ffmpeg(cmd_copy)
    if code == 0:
        return

    cmd_aac = [
        ffmpeg, "-y",
        "-ss", f"{start_sec:.6f}",
        "-to", f"{end_sec:.6f}",
        "-i", original_video,
        "-i", video_noaudio,
        "-map", "1:v:0",
        "-map", "0:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    code2, err2 = _run_ffmpeg(cmd_aac)
    if code2 == 0:
        return

    raise RuntimeError(f"Failed to mux audio.\n[copy]\n{err}\n\n[aac]\n{err2}\n")

# -------------------------
# SAM3 model/session helpers (app.py 鬨ｾ蛹・ｽｽ・ｱ髫ｴ螟ｲ・ｽ・･)
# -------------------------

MODEL_ID = "facebook/sam3"

_TRACKER_MODEL: Optional[Sam3TrackerVideoModel] = None
_TRACKER_PROCESSOR: Optional[Sam3TrackerVideoProcessor] = None

def _is_torch_compile_runtime_failure(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    if (
        "backendcompilerfailed" in text
        or "torch._dynamo" in text
        or "torch._inductor" in text
        or "inductor" in text
        or "dynamo" in text
        or "pythondispatcher" in text
        or "pythondispatchertls was not set" in text
    ):
        return True
    try:
        tb = traceback.format_exc().lower()
    except Exception:
        tb = ""
    return (
        "backendcompilerfailed" in tb
        or "torch._dynamo" in tb
        or "torch._inductor" in tb
        or "pythondispatcher" in tb
    )

def _disable_torch_compile_runtime(reason: str = "") -> None:
    global _JOB_USE_TORCH_COMPILE, _TRACKER_MODEL, _TRACKER_PROCESSOR
    if _JOB_USE_TORCH_COMPILE:
        msg = "[WARN] torch.compile disabled at runtime"
        if reason:
            msg = f"{msg}: {reason}"
        print(msg)
    _JOB_USE_TORCH_COMPILE = False
    _TRACKER_MODEL = None
    _TRACKER_PROCESSOR = None
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def _run_frame_inference_with_compile_fallback(state: "AppState", frame_idx: int):
    model, _ = ensure_models_loaded()
    sess_frame_idx = _global_to_session_frame_idx(state, int(frame_idx), strict=True)
    try:
        with torch.no_grad():
            return model(
                inference_session=state.inference_session,
                frame_idx=int(sess_frame_idx),
            )
    except Exception as e:
        if _JOB_USE_TORCH_COMPILE and _is_torch_compile_runtime_failure(e):
            _disable_torch_compile_runtime(str(e))
            model_retry, _ = ensure_models_loaded()
            with torch.no_grad():
                return model_retry(
                    inference_session=state.inference_session,
                    frame_idx=int(sess_frame_idx),
                )
        raise

def ensure_models_loaded() -> Tuple[Sam3TrackerVideoModel, Sam3TrackerVideoProcessor]:
    global _TRACKER_MODEL, _TRACKER_PROCESSOR, _JOB_USE_TORCH_COMPILE
    if _TRACKER_MODEL is None or _TRACKER_PROCESSOR is None:
        _configure_cuda_runtime()
        dev = torch.device(_JOB_DEVICE)
        model = Sam3TrackerVideoModel.from_pretrained(MODEL_ID, torch_dtype=_JOB_DTYPE).to(dev).eval()
        if _JOB_DEVICE == "cuda" and _JOB_USE_TORCH_COMPILE and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="max-autotune", fullgraph=False, dynamic=True)
            except Exception as e:
                _JOB_USE_TORCH_COMPILE = False
                print("[WARN] torch.compile disabled:", e)
        elif _JOB_DEVICE == "cuda" and not _JOB_USE_TORCH_COMPILE:
            print("[INFO] torch.compile is disabled for this environment.")
        _TRACKER_MODEL = model
        _TRACKER_PROCESSOR = Sam3TrackerVideoProcessor.from_pretrained(MODEL_ID)
    return _TRACKER_MODEL, _TRACKER_PROCESSOR

def _call_init_video_session(processor, **kwargs):
    sig = inspect.signature(processor.init_video_session)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return processor.init_video_session(**filtered)

def _call_add_inputs(processor, **kwargs):
    sig = inspect.signature(processor.add_inputs_to_inference_session)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return processor.add_inputs_to_inference_session(**filtered)

def _clear_points_for_object_all_frames(state: "AppState", obj_id: int) -> None:
    """
    Some SAM3 variants reject a box prompt if points already exist for the same object.
    Clear points for this object on all frames so UI cross markers and backend state match.
    """
    if state is None:
        return
    oid = int(obj_id)
    changed_frames = []
    for fidx, mp in list(state.clicks_by_frame_obj.items()):
        if not isinstance(mp, dict):
            continue
        if oid in mp and mp[oid]:
            mp[oid] = []
            changed_frames.append(int(fidx))
    # 鬮ｯ・ｦ繝ｻ・ｨ鬩穂ｼ夲ｽｽ・ｺ驛｢・ｧ繝ｻ・ｭ驛｢譎｢・ｽ・｣驛｢譏ｴ繝ｻ邵ｺ蜥擾ｽｹ譎｢・ｽ・･驛｢・ｧ陜｣・､隨冗霜諤上・・ｹ髯具ｽｹ郢晢ｽｻ
    for f in changed_frames:
        state.composited_frames.pop(f, None)

def _to_int_list(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], (list, tuple)):
        x = x[0]
    try:
        return [int(v) for v in x]
    except Exception:
        return None

def _normalize_masks(m, obj_ids=None) -> torch.Tensor:
    if isinstance(m, np.ndarray):
        m = torch.from_numpy(m)
    if not isinstance(m, torch.Tensor):
        raise gr.Error(f"mask must be Tensor/ndarray, got: {type(m)}")

    n = len(obj_ids) if obj_ids is not None else None

    if m.ndim == 4:
        if n is not None:
            if m.shape[0] == n and m.shape[1] == 1:
                m = m[:, 0]
            elif m.shape[0] == 1 and m.shape[1] == n:
                m = m[0]
            else:
                if m.shape[0] == 1:
                    m = m[0]
                elif m.shape[1] == 1:
                    m = m[:, 0]
                else:
                    raise gr.Error(f"Unexpected 4D mask shape: {tuple(m.shape)} (obj_ids={obj_ids})")
        else:
            if m.shape[0] == 1:
                m = m[0]
            elif m.shape[1] == 1:
                m = m[:, 0]
            else:
                raise gr.Error(f"Unexpected 4D mask shape: {tuple(m.shape)}")
    elif m.ndim == 3:
        pass
    else:
        raise gr.Error(f"Unexpected mask tensor shape: {tuple(m.shape)}")

    if m.ndim != 3:
        raise gr.Error(f"Normalize failed: got {tuple(m.shape)}")
    return m

def _mask_to_u8_2d(mask: Any, h: Optional[int] = None, w: Optional[int] = None) -> np.ndarray:
    mm = mask
    if isinstance(mm, tuple) and len(mm) == 3:
        try:
            mh, mw, packed = int(mm[0]), int(mm[1]), mm[2]
            if isinstance(packed, np.ndarray) and mh >= 0 and mw >= 0:
                count = mh * mw
                if count <= 0:
                    mm = np.zeros((0, 0), dtype=np.uint8)
                else:
                    flat = np.unpackbits(packed)
                    if flat.size < count:
                        flat = np.pad(flat, (0, count - flat.size), constant_values=0)
                    mm = flat[:count].reshape((mh, mw)).astype(np.uint8, copy=False)
        except Exception:
            pass
    if isinstance(mm, torch.Tensor):
        # Avoid direct bfloat16 -> numpy conversion (unsupported on some builds).
        t = mm.detach()
        try:
            t = (t > 0).to(torch.uint8)
        except Exception:
            t = t.to(torch.float32)
            t = (t > 0).to(torch.uint8)
        mm = t.cpu().numpy()
    mm = np.asarray(mm)
    if mm.ndim == 3:
        mm = mm.squeeze()
    if mm.ndim != 2:
        raise gr.Error(f"mask must be 2D, got shape={tuple(mm.shape)}")
    if mm.dtype != np.uint8:
        mm = (mm > 0).astype(np.uint8)
    else:
        mm = (mm > 0).astype(np.uint8, copy=False)
    if h is not None and w is not None and mm.shape != (int(h), int(w)):
        mm = cv2.resize(mm, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
        mm = (mm > 0).astype(np.uint8, copy=False)
    return mm

def _pack_mask_for_storage(mask: Any) -> Tuple[int, int, np.ndarray]:
    mm = _mask_to_u8_2d(mask)
    h, w = int(mm.shape[0]), int(mm.shape[1])
    packed = np.packbits(mm.reshape(-1))
    return (h, w, packed)

def pastel_color_for_object(obj_id: int) -> Tuple[int, int, int]:
    golden_ratio_conjugate = 0.61
    hue = (obj_id * golden_ratio_conjugate) % 1.0
    saturation = 0.45
    value = 1.0
    import colorsys
    r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r_f * 255), int(g_f * 255), int(b_f * 255)

def overlay_masks_on_frame(
    frame: Image.Image,
    masks_per_object: Dict[int, Any],
    color_by_obj: Dict[int, Tuple[int, int, int]],
    alpha: float = 0.55,
) -> Image.Image:
    base = np.array(frame).astype(np.float32) / 255.0
    overlay = base.copy()

    for obj_id, mask in masks_per_object.items():
        if mask is None:
            continue
        mm = _mask_to_u8_2d(mask, base.shape[0], base.shape[1]).astype(np.float32, copy=False)
        color = np.array(color_by_obj.get(obj_id, (255, 0, 0)), dtype=np.float32) / 255.0
        m = mm[..., None]
        overlay = (1.0 - alpha * m) * overlay + (alpha * m) * color

    out = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def _ensure_color_for_obj(color_by_obj: Dict[int, Tuple[int, int, int]], obj_id: int) -> None:
    if obj_id not in color_by_obj:
        color_by_obj[obj_id] = pastel_color_for_object(obj_id)

# -------------------------
# App state for job UI
# -------------------------

class AppState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.video_frames: Any = []
        self.video_fps: float = 0.0
        self.segment_start_sec: float = 0.0
        self.segment_end_sec: float = 0.0
        self.segment_video_path: str = ""
        self.segment_video_owned: bool = False
        self.session_chunk_frames: int = max(32, int(_JOB_SESSION_CHUNK_FRAMES))
        self.loaded_chunk_start_idx: int = -1
        self.loaded_chunk_end_idx: int = -1
        self.original_frame_size: Tuple[int, int] = (0, 0)  # (h, w)
        self.source_video_path: str = ""
        self.job_id: str = ""
        self.job_dir: str = ""
        self.output_dir: str = ""
        # last rendered preview mp4 for Gradio (H.264/AAC)
        self.preview_mp4_path: str = ""

        self.inference_session = None
        self.masks_by_frame: Dict[int, Dict[int, Tuple[int, int, np.ndarray]]] = {}
        self.color_by_obj: Dict[int, Tuple[int, int, int]] = {}
        self.clicks_by_frame_obj: Dict[int, Dict[int, List[Tuple[int,int,int]]]] = {}
        self.boxes_by_frame_obj: Dict[int, Dict[int, List[Tuple[int,int,int,int]]]] = {}
        self.composited_frames: "OrderedDict[int, Image.Image]" = OrderedDict()

        self.current_frame_idx: int = 0
        self.current_obj_id: int = 1
        self.current_label: str = "positive"
        self.current_clear_old: bool = False
        self.current_prompt_type: str = "Points"

        self.pending_box_start: Optional[Tuple[int,int]] = None
        self.pending_box_start_frame_idx: Optional[int] = None
        self.pending_box_start_obj_id: Optional[int] = None

    @property
    def num_frames(self) -> int:
        return len(self.video_frames)

_APP_STATE_REGISTRY: Dict[str, AppState] = {}
_APP_STATE_LOCK = threading.Lock()

def _register_app_state(job_key: str, state: AppState) -> None:
    k = str(job_key or "").strip()
    if not k:
        raise ValueError("job_key is empty")
    with _APP_STATE_LOCK:
        _APP_STATE_REGISTRY[k] = state

def _get_app_state(job_key: str) -> Optional[AppState]:
    k = str(job_key or "").strip()
    if not k:
        return None
    with _APP_STATE_LOCK:
        return _APP_STATE_REGISTRY.get(k)

def _release_app_state(state: Optional[AppState]) -> None:
    if state is None:
        return
    try:
        vf = getattr(state, "video_frames", None)
        if hasattr(vf, "clear"):
            vf.clear()
        state.video_frames = []
    except Exception:
        pass
    seg_path = ""
    seg_owned = False
    try:
        seg_path = str(getattr(state, "segment_video_path", "") or "")
        seg_owned = bool(getattr(state, "segment_video_owned", False))
        state.segment_video_path = ""
        state.segment_video_owned = False
    except Exception:
        pass
    if seg_owned and seg_path:
        _safe_remove_file(seg_path)
    try:
        state.masks_by_frame.clear()
        state.color_by_obj.clear()
        state.clicks_by_frame_obj.clear()
        state.boxes_by_frame_obj.clear()
        state.composited_frames.clear()
    except Exception:
        pass
    try:
        state.inference_session = None
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass

def _unregister_app_state(job_key: str) -> None:
    k = str(job_key or "").strip()
    if not k:
        return
    st: Optional[AppState] = None
    with _APP_STATE_LOCK:
        st = _APP_STATE_REGISTRY.pop(k, None)
    _release_app_state(st)

def _chunk_range_for_frame(total_frames: int, frame_idx: int, chunk_frames: int) -> Tuple[int, int]:
    total = max(0, int(total_frames))
    if total <= 0:
        return 0, 0
    c = max(1, int(chunk_frames))
    i = int(np.clip(int(frame_idx), 0, total - 1))
    s = (i // c) * c
    e = min(total, s + c)
    return s, e

def _chunk_retry_candidates(chunk_frames: int, min_chunk: int = 32) -> List[int]:
    c = max(int(min_chunk), int(chunk_frames))
    out: List[int] = [c]
    while c > int(min_chunk):
        c = max(int(min_chunk), c // 2)
        if c != out[-1]:
            out.append(c)
        if c == int(min_chunk):
            break
    return out

def _is_low_memory_error(err: Exception) -> bool:
    msg = str(err or "")
    msg_l = msg.lower()
    if "unable to allocate" in msg_l:
        return True
    needles = [
        "out of memory",
        "cuda out of memory",
        "cannot allocate memory",
        "not enough memory",
        "std::bad_alloc",
        "allocation failed",
    ]
    return any(n in msg_l for n in needles)

def _build_streaming_inference_session(processor: Sam3TrackerVideoProcessor):
    dev = torch.device(_JOB_DEVICE)
    kwargs = dict(
        video=None,
        inference_device=dev,
        processing_device=_JOB_SESSION_PROCESSING_DEVICE,
        video_storage_device=_JOB_SESSION_VIDEO_STORAGE_DEVICE,
        inference_state_device=_JOB_SESSION_INFERENCE_STATE_DEVICE,
        dtype=_JOB_DTYPE,
    )
    sess = _call_init_video_session(processor, **kwargs)
    if hasattr(sess, "inference_device"):
        sess.inference_device = dev
    if hasattr(sess, "cache") and hasattr(sess.cache, "inference_device"):
        sess.cache.inference_device = dev
    return sess

def _preprocess_frame_for_streaming(
    processor: Sam3TrackerVideoProcessor,
    frame: Image.Image,
) -> torch.Tensor:
    batch = processor.video_processor.preprocess(videos=[frame], return_tensors="pt")
    pv = batch["pixel_values_videos"][0, 0]  # (C,H,W)
    return pv

def _load_frame_chunk_into_session(
    state: AppState,
    processor: Sam3TrackerVideoProcessor,
    chunk_start: int,
    chunk_end: int,
) -> None:
    if state is None:
        raise RuntimeError("state is None")
    sess = state.inference_session
    if sess is None or not hasattr(sess, "add_new_frame"):
        raise RuntimeError("inference session is not initialized for streaming")
    total = max(0, int(chunk_end) - int(chunk_start))
    if total <= 0:
        return

    cs = int(chunk_start)
    for k, frame_idx in enumerate(range(int(chunk_start), int(chunk_end)), start=1):
        frame = state.video_frames[frame_idx]
        if frame_idx == int(chunk_start):
            w, h = frame.size
            state.original_frame_size = (int(h), int(w))
            # Streaming init may not set these fields; keep them valid for
            # post_process_masks callers that still reference session size.
            try:
                if hasattr(sess, "video_height"):
                    sess.video_height = int(h)
                if hasattr(sess, "video_width"):
                    sess.video_width = int(w)
            except Exception:
                pass
        pv = _preprocess_frame_for_streaming(processor, frame)
        # Session-local frame indices must start from 0 for each chunk.
        # SAM3 uses `num_frames=len(processed_frames)` internally.
        local_idx = int(frame_idx) - cs
        sess.add_new_frame(pv, frame_idx=int(local_idx))
        if (k % 90) == 0:
            gc.collect()

def _global_to_session_frame_idx(
    state: AppState,
    frame_idx: int,
    strict: bool = True,
) -> int:
    g = int(frame_idx)
    s = int(getattr(state, "loaded_chunk_start_idx", -1))
    e = int(getattr(state, "loaded_chunk_end_idx", -1))
    if s >= 0 and e > s:
        local = g - s
        if strict and not (0 <= local < (e - s)):
            raise RuntimeError(
                f"frame_idx out of loaded chunk: global={g}, chunk=[{s},{e})"
            )
        return int(local)
    return int(g)

def _session_to_global_frame_idx(state: AppState, session_frame_idx: int) -> int:
    s = int(getattr(state, "loaded_chunk_start_idx", -1))
    e = int(getattr(state, "loaded_chunk_end_idx", -1))
    if s >= 0 and e > s:
        return int(s + int(session_frame_idx))
    return int(session_frame_idx)

def _replay_prompts_for_chunk(
    state: AppState,
    processor: Sam3TrackerVideoProcessor,
    chunk_start: int,
    chunk_end: int,
) -> None:
    if state is None or state.inference_session is None:
        return
    h, w = state.original_frame_size
    if h <= 0 or w <= 0:
        first = state.video_frames[0]
        w, h = first.size
        state.original_frame_size = (int(h), int(w))

    for frame_idx in range(int(chunk_start), int(chunk_end)):
        clicks_map = state.clicks_by_frame_obj.get(frame_idx, {})
        boxes_map = state.boxes_by_frame_obj.get(frame_idx, {})
        obj_ids = sorted(set(clicks_map.keys()) | set(boxes_map.keys()))
        sess_frame_idx = _global_to_session_frame_idx(state, int(frame_idx), strict=True)
        for oid in obj_ids:
            boxes = boxes_map.get(oid, [])
            if boxes:
                x1, y1, x2, y2 = boxes[-1]
                box3 = [[[int(x1), int(y1), int(x2), int(y2)]]]
                _call_add_inputs(
                    processor,
                    inference_session=state.inference_session,
                    frame_idx=int(sess_frame_idx),
                    obj_ids=[int(oid)],
                    input_boxes=box3,
                    original_size=(int(h), int(w)),
                    # Box prompts require clear_old_inputs=True in transformers SAM3.
                    clear_old_inputs=True,
                )
                continue
            clicks = clicks_map.get(oid, [])
            if clicks:
                points = [[[[int(c[0]), int(c[1])] for c in clicks]]]
                labels = [[[int(c[2]) for c in clicks]]]
                _call_add_inputs(
                    processor,
                    inference_session=state.inference_session,
                    frame_idx=int(sess_frame_idx),
                    obj_ids=[int(oid)],
                    input_points=points,
                    input_labels=labels,
                    original_size=(int(h), int(w)),
                    # Replay full click list for this frame/object deterministically.
                    clear_old_inputs=True,
                )

def ensure_inference_session_for_frame(
    state: AppState,
    frame_idx: int,
    force_reload: bool = False,
) -> None:
    if state is None:
        raise RuntimeError("state is None")
    if state.num_frames <= 0:
        raise RuntimeError("No video frames are loaded.")

    fidx = int(np.clip(int(frame_idx), 0, state.num_frames - 1))
    base_chunk = max(32, int(getattr(state, "session_chunk_frames", _JOB_SESSION_CHUNK_FRAMES)))
    if (
        (not force_reload)
        and state.inference_session is not None
        and state.loaded_chunk_start_idx <= fidx < state.loaded_chunk_end_idx
    ):
        return

    _, processor = ensure_models_loaded()
    last_err: Optional[Exception] = None
    chunk_candidates = _chunk_retry_candidates(base_chunk, min_chunk=32)

    for i, candidate_chunk in enumerate(chunk_candidates):
        chunk_start, chunk_end = _chunk_range_for_frame(state.num_frames, fidx, candidate_chunk)

        # Release old chunk/session before loading a new one.
        state.inference_session = None
        state.loaded_chunk_start_idx = -1
        state.loaded_chunk_end_idx = -1
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            state.inference_session = _build_streaming_inference_session(processor)
            _load_frame_chunk_into_session(state, processor, chunk_start, chunk_end)
            state.loaded_chunk_start_idx = int(chunk_start)
            state.loaded_chunk_end_idx = int(chunk_end)
            _replay_prompts_for_chunk(state, processor, chunk_start, chunk_end)
            if int(state.session_chunk_frames) != int(candidate_chunk):
                print(
                    "[WARN] Session chunk size adjusted:",
                    f"{state.session_chunk_frames} -> {candidate_chunk}",
                )
            state.session_chunk_frames = int(candidate_chunk)
            return
        except Exception as e:
            last_err = e
            state.inference_session = None
            state.loaded_chunk_start_idx = -1
            state.loaded_chunk_end_idx = -1
            has_next = (i + 1) < len(chunk_candidates)
            if _is_low_memory_error(e) and has_next:
                next_chunk = chunk_candidates[i + 1]
                print(
                    "[WARN] Chunk session load failed by memory pressure.",
                    f"retry chunk={next_chunk} (prev={candidate_chunk})",
                )
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise

    if last_err is not None:
        raise last_err

def _union_mask_for_frame(state: AppState, frame_idx: int, h: int, w: int) -> np.ndarray:
    masks = state.masks_by_frame.get(frame_idx, {})
    union = None
    for m in masks.values():
        if m is None:
            continue
        mm = _mask_to_u8_2d(m, h, w)
        union = mm if union is None else (union | mm)
    if union is None:
        union = np.zeros((h, w), dtype=np.uint8)
    return union.astype(bool)

def compose_frame(state: AppState, frame_idx: int) -> Image.Image:
    frame_idx = int(np.clip(frame_idx, 0, state.num_frames - 1))
    frame = state.video_frames[frame_idx]
    out_img = frame.copy()

    masks = state.masks_by_frame.get(frame_idx, {})
    if masks:
        out_img = overlay_masks_on_frame(out_img, masks, state.color_by_obj, alpha=0.65)

    # clicks overlay
    clicks_map = state.clicks_by_frame_obj.get(frame_idx)
    if clicks_map:
        draw = ImageDraw.Draw(out_img)
        cross_half = 6
        for obj_id, pts in clicks_map.items():
            for x, y, lbl in pts:
                color = (0, 255, 0) if int(lbl) == 1 else (255, 0, 0)
                draw.line([(x - cross_half, y), (x + cross_half, y)], fill=color, width=2)
                draw.line([(x, y - cross_half), (x, y + cross_half)], fill=color, width=2)

    # pending box start marker
    if (
        state.pending_box_start is not None
        and state.pending_box_start_frame_idx == frame_idx
        and state.pending_box_start_obj_id is not None
    ):
        draw = ImageDraw.Draw(out_img)
        x, y = state.pending_box_start
        cross_half = 6
        color = state.color_by_obj.get(state.pending_box_start_obj_id, (255, 255, 255))
        draw.line([(x - cross_half, y), (x + cross_half, y)], fill=color, width=2)
        draw.line([(x, y - cross_half), (x, y + cross_half)], fill=color, width=2)

    # boxes overlay
    box_map = state.boxes_by_frame_obj.get(frame_idx)
    if box_map:
        draw = ImageDraw.Draw(out_img)
        for obj_id, boxes in box_map.items():
            color = state.color_by_obj.get(obj_id, (255, 255, 255))
            for x1, y1, x2, y2 in boxes:
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

    # Cap preview cache so long scrubbing does not keep every composited frame.
    if frame_idx in state.composited_frames:
        state.composited_frames.pop(frame_idx, None)
    state.composited_frames[frame_idx] = out_img
    while len(state.composited_frames) > _COMPOSITED_CACHE_MAX_FRAMES:
        state.composited_frames.popitem(last=False)
    return out_img

def update_frame_display(state: AppState, frame_idx: int) -> Image.Image:
    frame_idx = int(np.clip(frame_idx, 0, state.num_frames - 1))
    cached = state.composited_frames.get(frame_idx)
    if cached is not None:
        return cached
    return compose_frame(state, frame_idx)

def reset_prompts_keep_video(state: AppState) -> None:
    """
    Equivalent to app.py reset_session.
    - Keep video data (frames/fps/segment/job metadata)
    - Clear prompt-derived state (clicks/boxes/masks/cache/colors)
    - Reset loaded session state; next interaction lazily rebuilds the chunk session.
    """
    if state is None:
        return
    if state.num_frames <= 0:
        return

    try:
        if state.inference_session is not None and hasattr(state.inference_session, "reset_inference_session"):
            state.inference_session.reset_inference_session()
        else:
            state.inference_session = None
    except Exception:
        traceback.print_exc()
        state.inference_session = None

    state.loaded_chunk_start_idx = -1
    state.loaded_chunk_end_idx = -1

    state.masks_by_frame.clear()
    state.clicks_by_frame_obj.clear()
    state.boxes_by_frame_obj.clear()
    state.composited_frames.clear()
    state.color_by_obj.clear()

    state.pending_box_start = None
    state.pending_box_start_frame_idx = None
    state.pending_box_start_obj_id = None
    gc.collect()

def on_image_click(
    img: Image.Image,
    state: AppState,
    frame_idx: int,
    obj_id: int,
    label: str,
    clear_old: bool,
    prompt_type: str,
    evt: gr.SelectData,
) -> Tuple[Image.Image, AppState]:
    if state is None:
        return img, state

    # click coordinate
    x = y = None
    if evt is not None:
        try:
            if hasattr(evt, "index") and isinstance(evt.index, (list, tuple)) and len(evt.index) == 2:
                x, y = int(evt.index[0]), int(evt.index[1])
            elif hasattr(evt, "value") and isinstance(evt.value, dict) and "x" in evt.value and "y" in evt.value:
                x, y = int(evt.value["x"]), int(evt.value["y"])
        except Exception:
            x = y = None
    if x is None or y is None:
        raise gr.Error("Could not read click coordinates.")

    ann_frame_idx = int(frame_idx)
    ann_obj_id = int(obj_id)
    try:
        ensure_inference_session_for_frame(state, ann_frame_idx)
    except Exception as e:
        raise gr.Error(f"Failed to load frame chunk: {e}")
    if state.inference_session is None:
        return img, state

    _, processor = ensure_models_loaded()
    ann_sess_frame_idx = _global_to_session_frame_idx(state, ann_frame_idx, strict=True)
    h, w = state.original_frame_size
    if h <= 0 or w <= 0:
        fw, fh = state.video_frames[ann_frame_idx].size
        h, w = int(fh), int(fw)
        state.original_frame_size = (h, w)
    orig_size = (int(h), int(w))

    _ensure_color_for_obj(state.color_by_obj, ann_obj_id)

    prompt_type_norm = str(prompt_type).strip().lower()
    if prompt_type_norm in ("box", "boxes"):
        # 2-click box
        if (
            state.pending_box_start is not None
            and (
                state.pending_box_start_frame_idx != ann_frame_idx
                or state.pending_box_start_obj_id != ann_obj_id
            )
        ):
            # Do not pair box corners across different frame/object contexts.
            state.pending_box_start = None
            state.pending_box_start_frame_idx = None
            state.pending_box_start_obj_id = None
        if state.pending_box_start is None:
            frame_clicks = state.clicks_by_frame_obj.setdefault(ann_frame_idx, {})
            frame_clicks[ann_obj_id] = []
            state.composited_frames.pop(ann_frame_idx, None)
            state.pending_box_start = (int(x), int(y))
            state.pending_box_start_frame_idx = ann_frame_idx
            state.pending_box_start_obj_id = ann_obj_id
            return update_frame_display(state, ann_frame_idx), state

        x1, y1 = state.pending_box_start
        x2, y2 = int(x), int(y)
        state.pending_box_start = None
        state.pending_box_start_frame_idx = None
        state.pending_box_start_obj_id = None

        x_min, y_min = min(x1, x2), min(y1, y2)
        x_max, y_max = max(x1, x2), max(y1, y2)

        box3 = [[[x_min, y_min, x_max, y_max]]]
        _clear_points_for_object_all_frames(state, ann_obj_id)
        try:
            _call_add_inputs(
                processor,
                inference_session=state.inference_session,
                frame_idx=int(ann_sess_frame_idx),
                obj_ids=[ann_obj_id],
                input_boxes=box3,
                original_size=orig_size,
                clear_old_points=True,
                clear_old_boxes=True,
                clear_old_inputs=True,
            )
        except ValueError as e:
            msg = str(e)
            if "nested list with 4 levels" in msg or "Got 3 levels" in msg:
                box4 = [box3]  # [[[[x1,y1,x2,y2]]]]
                _call_add_inputs(
                    processor,
                    inference_session=state.inference_session,
                    frame_idx=int(ann_sess_frame_idx),
                    obj_ids=[ann_obj_id],
                    input_boxes=box4,
                    original_size=orig_size,
                    clear_old_points=True,
                    clear_old_boxes=True,
                    clear_old_inputs=True,
                )
            else:
                raise
        except Exception as e:
            if _JOB_USE_TORCH_COMPILE and _is_torch_compile_runtime_failure(e):
                _disable_torch_compile_runtime(str(e))
                raise gr.Error("Torch runtime error detected. torch.compile was disabled. Please click again.")
            raise

        frame_boxes = state.boxes_by_frame_obj.setdefault(ann_frame_idx, {})
        obj_boxes = frame_boxes.setdefault(ann_obj_id, [])
        obj_boxes.clear()
        obj_boxes.append((x_min, y_min, x_max, y_max))
        state.composited_frames.pop(ann_frame_idx, None)

    else:
        label_int = 1 if str(label).lower().startswith("pos") else 0
        frame_clicks = state.clicks_by_frame_obj.setdefault(ann_frame_idx, {})
        obj_clicks = frame_clicks.setdefault(ann_obj_id, [])

        if bool(clear_old):
            obj_clicks.clear()
            frame_boxes = state.boxes_by_frame_obj.setdefault(ann_frame_idx, {})
            frame_boxes[ann_obj_id] = []

        obj_clicks.append((int(x), int(y), int(label_int)))

        # Send only the new click as delta input. Sending the full click list with
        # clear_old_inputs=False can duplicate past clicks inside the tracker.
        points = [[[[int(x), int(y)]]]]
        labels = [[[int(label_int)]]]

        try:
            _call_add_inputs(
                processor,
                inference_session=state.inference_session,
                frame_idx=int(ann_sess_frame_idx),
                obj_ids=[ann_obj_id],
                input_points=points,
                input_labels=labels,
                original_size=orig_size,
                clear_old_inputs=bool(clear_old),
            )
        except Exception as e:
            if _JOB_USE_TORCH_COMPILE and _is_torch_compile_runtime_failure(e):
                _disable_torch_compile_runtime(str(e))
                raise gr.Error("Torch runtime error detected. torch.compile was disabled. Please click again.")
            raise
        state.composited_frames.pop(ann_frame_idx, None)

    # run inference for that frame
    outputs = _run_frame_inference_with_compile_fallback(state, ann_frame_idx)

    try:
        processed = processor.post_process_masks(
            [outputs.pred_masks],
            [[state.inference_session.video_height, state.inference_session.video_width]],
            binarize=False,
        )[0]
    except Exception as e:
        if _JOB_USE_TORCH_COMPILE and _is_torch_compile_runtime_failure(e):
            _disable_torch_compile_runtime(str(e))
            raise gr.Error("Torch runtime error detected. torch.compile was disabled. Please click again.")
        raise

    out_obj_ids = _to_int_list(getattr(outputs, "object_ids", None))
    if out_obj_ids is None:
        out_obj_ids = [ann_obj_id]

    masks = _normalize_masks(processed, out_obj_ids)  # (K,H,W)
    K = masks.shape[0]
    if len(out_obj_ids) != K:
        out_obj_ids = out_obj_ids[:K]

    masks_for_frame = state.masks_by_frame.setdefault(ann_frame_idx, {})
    for k, oid in enumerate(out_obj_ids):
        _ensure_color_for_obj(state.color_by_obj, int(oid))
        masks_for_frame[int(oid)] = _pack_mask_for_storage(masks[k])

    state.composited_frames.pop(ann_frame_idx, None)
    return update_frame_display(state, ann_frame_idx), state

# -------------------------
# Propagate
# -------------------------

def _collect_prompt_frame_indices(state: AppState) -> List[int]:
    out: set[int] = set()
    for fidx, mp in state.clicks_by_frame_obj.items():
        if not isinstance(mp, dict):
            continue
        if any(bool(v) for v in mp.values()):
            out.add(int(fidx))
    for fidx, mp in state.boxes_by_frame_obj.items():
        if not isinstance(mp, dict):
            continue
        if any(bool(v) for v in mp.values()):
            out.add(int(fidx))
    return sorted([f for f in out if 0 <= int(f) < int(state.num_frames)])

def _bbox_from_mask_storage(mask_data: Any, h: int, w: int, pad: int = 2) -> Optional[Tuple[int, int, int, int]]:
    mm = _mask_to_u8_2d(mask_data, int(h), int(w))
    ys, xs = np.where(mm > 0)
    if ys.size == 0 or xs.size == 0:
        return None

    x1 = max(0, int(xs.min()) - int(pad))
    y1 = max(0, int(ys.min()) - int(pad))
    x2 = min(int(w) - 1, int(xs.max()) + int(pad))
    y2 = min(int(h) - 1, int(ys.max()) + int(pad))
    if x2 <= x1:
        x2 = min(int(w) - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(int(h) - 1, y1 + 1)
    return (int(x1), int(y1), int(x2), int(y2))

def _seed_boxes_from_frame_masks(state: AppState, frame_idx: int, h: int, w: int) -> Dict[int, Tuple[int, int, int, int]]:
    out: Dict[int, Tuple[int, int, int, int]] = {}
    frame_masks = state.masks_by_frame.get(int(frame_idx), {})
    if not isinstance(frame_masks, dict):
        return out
    for oid, m in frame_masks.items():
        box = _bbox_from_mask_storage(m, int(h), int(w))
        if box is not None:
            out[int(oid)] = box
    return out

def _seed_boxes_from_chunk_scan(
    state: AppState,
    chunk_start: int,
    chunk_end: int,
    h: int,
    w: int,
    prefer_end: bool,
) -> Dict[int, Tuple[int, int, int, int]]:
    cs = int(chunk_start)
    ce = int(chunk_end)
    if ce <= cs:
        return {}
    if prefer_end:
        it = range(ce - 1, cs - 1, -1)
    else:
        it = range(cs, ce)
    for fidx in it:
        seed = _seed_boxes_from_frame_masks(state, fidx, int(h), int(w))
        if seed:
            return seed
    return {}

def _inject_seed_boxes(
    state: AppState,
    processor: Sam3TrackerVideoProcessor,
    frame_idx: int,
    seed_boxes: Dict[int, Tuple[int, int, int, int]],
    original_size: Tuple[int, int],
    seed_frame_idx: Optional[int] = None,
    mask_h: Optional[int] = None,
    mask_w: Optional[int] = None,
) -> None:
    if not seed_boxes or state.inference_session is None:
        return
    oh, ow = int(original_size[0]), int(original_size[1])
    injected_obj_ids: set[int] = set()
    sess_frame_idx = _global_to_session_frame_idx(state, int(frame_idx), strict=True)

    # Prefer seeding with the actual previous/next-frame masks when available.
    if (
        seed_frame_idx is not None
        and mask_h is not None
        and mask_w is not None
        and 0 <= int(seed_frame_idx) < int(state.num_frames)
    ):
        src_masks = state.masks_by_frame.get(int(seed_frame_idx), {})
        if isinstance(src_masks, dict):
            for oid, packed in src_masks.items():
                try:
                    mm = _mask_to_u8_2d(packed, int(mask_h), int(mask_w))
                    if int(np.count_nonzero(mm)) <= 0:
                        continue
                    _call_add_inputs(
                        processor,
                        inference_session=state.inference_session,
                        frame_idx=int(sess_frame_idx),
                        obj_ids=[int(oid)],
                        input_masks=[mm.astype(bool)],
                    )
                    injected_obj_ids.add(int(oid))
                except Exception as e:
                    print(
                        f"[WARN] Seed mask injection failed for obj={oid} "
                        f"src_frame={seed_frame_idx} dst_frame={frame_idx}: {e}"
                    )

    for oid, (x1, y1, x2, y2) in seed_boxes.items():
        if int(oid) in injected_obj_ids:
            continue
        try:
            _call_add_inputs(
                processor,
                inference_session=state.inference_session,
                frame_idx=int(sess_frame_idx),
                obj_ids=[int(oid)],
                input_boxes=[[[int(x1), int(y1), int(x2), int(y2)]]],
                original_size=(oh, ow),
                # Box prompts require clear_old_inputs=True in transformers SAM3.
                clear_old_inputs=True,
            )
        except Exception as e:
            print(f"[WARN] Seed box injection failed for obj={oid} frame={frame_idx}: {e}")

def _ensure_full_session_for_propagation(
    state: AppState,
    processor: Sam3TrackerVideoProcessor,
) -> None:
    if state is None:
        raise RuntimeError("state is None")
    total = int(state.num_frames)
    if total <= 0:
        raise RuntimeError("No frames are loaded.")

    already_full = (
        state.inference_session is not None
        and int(getattr(state, "loaded_chunk_start_idx", -1)) == 0
        and int(getattr(state, "loaded_chunk_end_idx", -1)) >= total
    )
    if already_full:
        return

    state.inference_session = None
    state.loaded_chunk_start_idx = -1
    state.loaded_chunk_end_idx = -1
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    state.inference_session = _build_streaming_inference_session(processor)
    state.loaded_chunk_start_idx = 0
    state.loaded_chunk_end_idx = int(total)
    _load_frame_chunk_into_session(state, processor, 0, int(total))
    _replay_prompts_for_chunk(state, processor, 0, int(total))

def _propagate_masks_full_session_legacy(
    state: AppState,
    model: Sam3TrackerVideoModel,
    processor: Sam3TrackerVideoProcessor,
    prompt_frames: List[int],
) -> Iterator[Tuple[AppState, str, Any, Image.Image]]:
    total = max(1, int(state.num_frames))
    seen_frames: set[int] = set()
    masked_frames: set[int] = set()

    state.current_frame_idx = int(min(prompt_frames))
    last_frame_idx = int(state.current_frame_idx)
    h, w = state.original_frame_size
    if h <= 0 or w <= 0:
        fw, fh = state.video_frames[last_frame_idx].size
        h, w = int(fh), int(fw)
        state.original_frame_size = (h, w)

    _ensure_full_session_for_propagation(state, processor)

    write_status(state.job_dir, state="running", phase="propagate", progress=0.35, message="Propagating...")
    yield state, f"Propagating masks: 0/{total}", gr.update(value=last_frame_idx), update_frame_display(state, last_frame_idx)

    start_frame_idx = int(min(prompt_frames))

    def _iter_outputs(reverse: bool):
        kwargs = {"inference_session": state.inference_session}
        sig = inspect.signature(model.propagate_in_video_iterator)
        if "reverse" in sig.parameters:
            kwargs["reverse"] = bool(reverse)
        if "start_frame_idx" in sig.parameters:
            kwargs["start_frame_idx"] = int(start_frame_idx)
        return model.propagate_in_video_iterator(**kwargs)

    def _consume(reverse: bool):
        nonlocal last_frame_idx
        for out in _iter_outputs(reverse=reverse):
            frame_idx = int(out.frame_idx)
            if frame_idx < 0 or frame_idx >= int(state.num_frames):
                continue

            video_res_masks = processor.post_process_masks(
                [out.pred_masks],
                original_sizes=[[int(h), int(w)]],
            )[0]
            masks3 = _normalize_masks(video_res_masks)
            out_obj_ids = _to_int_list(getattr(out, "object_ids", None))
            if out_obj_ids is None or len(out_obj_ids) != masks3.shape[0]:
                sess_obj_ids = [int(x) for x in getattr(state.inference_session, "obj_ids", [])]
                out_obj_ids = sess_obj_ids[: masks3.shape[0]]

            masks_for_frame = state.masks_by_frame.setdefault(frame_idx, {})
            for i, oid in enumerate(out_obj_ids):
                _ensure_color_for_obj(state.color_by_obj, int(oid))
                masks_for_frame[int(oid)] = _pack_mask_for_storage(masks3[i])
            state.composited_frames.pop(frame_idx, None)

            last_frame_idx = int(frame_idx)
            seen_frames.add(int(frame_idx))
            if len(masks_for_frame) > 0:
                masked_frames.add(int(frame_idx))

            covered = len(seen_frames)
            if covered % 20 == 0 or covered >= total:
                prog = 0.35 + 0.25 * (covered / float(total))
                write_status(
                    state.job_dir,
                    state="running",
                    phase="propagate",
                    progress=prog,
                    message=f"Propagating... {covered}/{total}",
                )
                state.current_frame_idx = int(last_frame_idx)
                yield (
                    state,
                    f"Propagating masks: {covered}/{total}",
                    gr.update(value=last_frame_idx),
                    update_frame_display(state, last_frame_idx),
                )

    with torch.no_grad():
        for item in _consume(reverse=False):
            yield item
        for item in _consume(reverse=True):
            yield item

    covered = len(seen_frames)
    masked = len(masked_frames)
    state.current_frame_idx = int(np.clip(last_frame_idx, 0, max(0, state.num_frames - 1)))
    done_msg = f"Propagation done. masked={masked}/{total}, visited={covered}/{total}"
    write_status(state.job_dir, state="running", phase="propagate", progress=0.60, message=done_msg)
    yield (
        state,
        done_msg,
        gr.update(value=state.current_frame_idx),
        update_frame_display(state, state.current_frame_idx),
    )

def propagate_masks(state: AppState) -> Iterator[Tuple[AppState, str, Any, Image.Image]]:
    if state is None:
        yield state, "Load failed.", gr.update(), None
        return
    if state.num_frames <= 0:
        yield state, "Load failed: no frames.", gr.update(), None
        return

    model, processor = ensure_models_loaded()

    total = max(1, int(state.num_frames))
    visited_frames: set[int] = set()
    masked_frames: set[int] = set()
    prompt_frames = _collect_prompt_frame_indices(state)

    # No prompts means there is no reliable tracking seed.
    if not prompt_frames:
        msg = "No prompts found. Add click/box prompt first."
        write_status(state.job_dir, state="running", phase="propagate", progress=0.35, message=msg)
        yield state, msg, gr.update(value=state.current_frame_idx), update_frame_display(state, state.current_frame_idx)
        return

    # Primary path: legacy full-session propagation (same behavior as the
    # pre-load-improvement implementation that users validated).
    try:
        for item in _propagate_masks_full_session_legacy(state, model, processor, prompt_frames):
            yield item
        return
    except Exception as e:
        if _JOB_USE_TORCH_COMPILE and _is_torch_compile_runtime_failure(e):
            _disable_torch_compile_runtime(str(e))
            model_retry, processor_retry = ensure_models_loaded()
            for item in _propagate_masks_full_session_legacy(state, model_retry, processor_retry, prompt_frames):
                yield item
            return
        print(f"[WARN] Full-session propagation failed; falling back to chunk mode: {e}")

    # Start from the first prompted frame to preserve user intent.
    state.current_frame_idx = int(min(prompt_frames))
    last_frame_idx = int(state.current_frame_idx)
    h, w = state.original_frame_size
    if h <= 0 or w <= 0:
        fw, fh = state.video_frames[last_frame_idx].size
        h, w = int(fh), int(fw)
        state.original_frame_size = (h, w)
    orig_size = (int(h), int(w))

    orig_session_chunk_frames = max(32, int(getattr(state, "session_chunk_frames", _JOB_SESSION_CHUNK_FRAMES)))
    prop_chunk_cap = max(32, int(_JOB_PROPAGATE_CHUNK_MAX_FRAMES))
    chunk_size = max(32, min(orig_session_chunk_frames, prop_chunk_cap))
    if chunk_size != orig_session_chunk_frames:
        print(
            "[INFO] Propagate chunk size override:",
            f"session={orig_session_chunk_frames} -> propagate={chunk_size}",
        )
    state.session_chunk_frames = int(chunk_size)

    prompt_frame_set = set(int(x) for x in prompt_frames)
    anchor_frame_idx = int(min(prompt_frames))
    anchor_chunk_start = (int(anchor_frame_idx) // int(chunk_size)) * int(chunk_size)
    anchor_chunk_end = min(int(state.num_frames), int(anchor_chunk_start) + int(chunk_size))

    write_status(state.job_dir, state="running", phase="propagate", progress=0.35, message="Propagating...")
    yield state, f"Propagating masks: 0/{total}", gr.update(value=last_frame_idx), update_frame_display(state, last_frame_idx)

    def _iter_outputs(start_frame_idx: int, max_track: int, reverse: bool):
        start_local = _global_to_session_frame_idx(state, int(start_frame_idx), strict=True)
        kwargs = {
            "inference_session": state.inference_session,
            "start_frame_idx": int(start_local),
            "max_frame_num_to_track": int(max_track),
            "reverse": bool(reverse),
        }
        sig = inspect.signature(model.propagate_in_video_iterator)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return model.propagate_in_video_iterator(**filtered)

    def _consume_chunk(chunk_start: int, chunk_end: int, start_idx: int, reverse: bool):
        nonlocal last_frame_idx
        if state.inference_session is None:
            return
        if bool(reverse):
            # reverse=True tracks from start_idx down to max(start_idx - max_track, 0)
            # Keep the iterator strictly inside [chunk_start, chunk_end).
            max_track = max(0, int(start_idx) - int(chunk_start))
        else:
            # reverse=False tracks from start_idx up to min(start_idx + max_track, num_frames-1)
            # Keep the iterator strictly inside [chunk_start, chunk_end).
            max_track = max(0, int(chunk_end) - int(start_idx) - 1)
        for out in _iter_outputs(start_frame_idx=int(start_idx), max_track=max_track, reverse=bool(reverse)):
            frame_idx = _session_to_global_frame_idx(state, int(out.frame_idx))
            if frame_idx < int(chunk_start) or frame_idx >= int(chunk_end):
                continue

            video_res_masks = processor.post_process_masks(
                [out.pred_masks],
                original_sizes=[[int(h), int(w)]],
            )[0]
            masks3 = _normalize_masks(video_res_masks)
            out_obj_ids = _to_int_list(getattr(out, "object_ids", None))
            if out_obj_ids is None or len(out_obj_ids) != masks3.shape[0]:
                sess_obj_ids = [int(x) for x in getattr(state.inference_session, "obj_ids", [])]
                out_obj_ids = sess_obj_ids[: masks3.shape[0]]

            masks_for_frame = state.masks_by_frame.setdefault(frame_idx, {})
            clicks_map = state.clicks_by_frame_obj.get(frame_idx, {})
            boxes_map = state.boxes_by_frame_obj.get(frame_idx, {})
            for i, oid in enumerate(out_obj_ids):
                has_manual_prompt = bool(clicks_map.get(int(oid))) or bool(boxes_map.get(int(oid)))
                # Keep user-confirmed prompt masks stable on prompt frames.
                if has_manual_prompt and int(oid) in masks_for_frame:
                    continue
                _ensure_color_for_obj(state.color_by_obj, int(oid))
                masks_for_frame[int(oid)] = _pack_mask_for_storage(masks3[i])
            state.composited_frames.pop(frame_idx, None)

            last_frame_idx = int(frame_idx)
            visited_frames.add(int(frame_idx))
            if len(masks_for_frame) > 0:
                masked_frames.add(int(frame_idx))

            covered = len(visited_frames)
            if covered % 20 == 0 or covered >= total:
                prog = 0.35 + 0.25 * (covered / float(total))
                write_status(
                    state.job_dir,
                    state="running",
                    phase="propagate",
                    progress=prog,
                    message=f"Propagating... {covered}/{total}",
                )
                state.current_frame_idx = last_frame_idx
                yield (
                    state,
                    f"Propagating masks: {covered}/{total}",
                    gr.update(value=last_frame_idx),
                    update_frame_display(state, last_frame_idx),
                )

    def _chunk_prompt_frames(chunk_start: int, chunk_end: int) -> List[int]:
        return [f for f in prompt_frame_set if int(chunk_start) <= int(f) < int(chunk_end)]

    def _find_seed_near_chunk(chunk_start: int, chunk_end: int) -> Tuple[Dict[int, Tuple[int, int, int, int]], Optional[int]]:
        # Search one chunk to the left first, then one chunk to the right.
        left_start = max(0, int(chunk_start) - int(chunk_size))
        left_end = int(chunk_start)
        if left_end > left_start:
            left_seed = _seed_boxes_from_chunk_scan(
                state,
                chunk_start=left_start,
                chunk_end=left_end,
                h=int(h),
                w=int(w),
                prefer_end=True,
            )
            if left_seed:
                return left_seed, int(left_end - 1)

        right_start = int(chunk_end)
        right_end = min(int(state.num_frames), int(chunk_end) + int(chunk_size))
        if right_end > right_start:
            right_seed = _seed_boxes_from_chunk_scan(
                state,
                chunk_start=right_start,
                chunk_end=right_end,
                h=int(h),
                w=int(w),
                prefer_end=False,
            )
            if right_seed:
                return right_seed, int(right_start)

        return {}, None

    def _mark_chunk_empty(chunk_start: int, chunk_end: int):
        nonlocal last_frame_idx
        for frame_idx in range(int(chunk_start), int(chunk_end)):
            if frame_idx in visited_frames:
                continue
            visited_frames.add(frame_idx)
            last_frame_idx = int(frame_idx)
            covered = len(visited_frames)
            if covered % 20 == 0 or covered >= total:
                prog = 0.35 + 0.25 * (covered / float(total))
                write_status(
                    state.job_dir,
                    state="running",
                    phase="propagate",
                    progress=prog,
                    message=f"Propagating... {covered}/{total}",
                )
                state.current_frame_idx = last_frame_idx
                yield (
                    state,
                    f"Propagating masks: {covered}/{total}",
                    gr.update(value=last_frame_idx),
                    update_frame_display(state, last_frame_idx),
                )

    try:
        with torch.no_grad():
            # Forward sweep from the first prompt chunk to the end.
            # Use the actually loaded chunk bounds because memory pressure may shrink
            # session chunk size dynamically.
            carry_seed_boxes: Dict[int, Tuple[int, int, int, int]] = {}
            first_forward_window: Optional[Tuple[int, int]] = None
            fwd_cursor = int(anchor_chunk_start)
            while fwd_cursor < int(state.num_frames):
                ensure_inference_session_for_frame(state, fwd_cursor, force_reload=True)
                if state.inference_session is None:
                    fwd_cursor += max(1, int(getattr(state, "session_chunk_frames", chunk_size)))
                    continue

                loaded_s = int(getattr(state, "loaded_chunk_start_idx", fwd_cursor))
                loaded_e = int(getattr(state, "loaded_chunk_end_idx", fwd_cursor + 1))
                chunk_start = max(int(fwd_cursor), int(loaded_s))
                chunk_end = min(int(state.num_frames), int(loaded_e))
                if chunk_end <= chunk_start:
                    fwd_cursor += max(1, int(getattr(state, "session_chunk_frames", chunk_size)))
                    continue

                if first_forward_window is None:
                    first_forward_window = (int(chunk_start), int(chunk_end))

                local_prompt_frames = _chunk_prompt_frames(chunk_start, chunk_end)
                has_local_prompt = len(local_prompt_frames) > 0

                if (not has_local_prompt) and carry_seed_boxes:
                    _inject_seed_boxes(
                        state,
                        processor,
                        frame_idx=chunk_start,
                        seed_boxes=carry_seed_boxes,
                        original_size=orig_size,
                        seed_frame_idx=(chunk_start - 1 if chunk_start > 0 else None),
                        mask_h=int(h),
                        mask_w=int(w),
                    )

                if has_local_prompt:
                    start_fwd = int(min(local_prompt_frames))
                    for item in _consume_chunk(chunk_start, chunk_end, start_idx=start_fwd, reverse=False):
                        yield item
                    # Fill earlier frames inside this chunk.
                    for item in _consume_chunk(chunk_start, chunk_end, start_idx=start_fwd, reverse=True):
                        yield item
                elif carry_seed_boxes:
                    for item in _consume_chunk(chunk_start, chunk_end, start_idx=chunk_start, reverse=False):
                        yield item
                else:
                    near_seed_boxes, near_seed_frame = _find_seed_near_chunk(chunk_start, chunk_end)
                    if near_seed_boxes:
                        _inject_seed_boxes(
                            state,
                            processor,
                            frame_idx=chunk_start,
                            seed_boxes=near_seed_boxes,
                            original_size=orig_size,
                            seed_frame_idx=near_seed_frame,
                            mask_h=int(h),
                            mask_w=int(w),
                        )
                        for item in _consume_chunk(chunk_start, chunk_end, start_idx=chunk_start, reverse=False):
                            yield item
                    else:
                        for item in _mark_chunk_empty(chunk_start, chunk_end):
                            yield item

                # Safety net: if model iterator skipped frames, mark uncovered frames as empty.
                for item in _mark_chunk_empty(chunk_start, chunk_end):
                    yield item

                next_seed = _seed_boxes_from_chunk_scan(
                    state,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    h=int(h),
                    w=int(w),
                    prefer_end=True,
                )
                if next_seed:
                    carry_seed_boxes = next_seed
                elif carry_seed_boxes:
                    # Keep last valid seed across temporary empty boundary frames.
                    pass

                fwd_cursor = int(chunk_end)

            # Backward sweep from just before the first prompt chunk to frame 0.
            if first_forward_window is not None:
                back_seed_start, back_seed_end = first_forward_window
            else:
                back_seed_start, back_seed_end = int(anchor_chunk_start), int(anchor_chunk_end)
            carry_seed_boxes_back = _seed_boxes_from_chunk_scan(
                state,
                chunk_start=int(back_seed_start),
                chunk_end=int(back_seed_end),
                h=int(h),
                w=int(w),
                prefer_end=False,
            )

            back_cursor = int(anchor_chunk_start) - 1
            while back_cursor >= 0:
                ensure_inference_session_for_frame(state, back_cursor, force_reload=True)
                if state.inference_session is None:
                    back_cursor -= max(1, int(getattr(state, "session_chunk_frames", chunk_size)))
                    continue

                loaded_s = int(getattr(state, "loaded_chunk_start_idx", back_cursor))
                loaded_e = int(getattr(state, "loaded_chunk_end_idx", back_cursor + 1))
                chunk_start = max(0, int(loaded_s))
                chunk_end = min(int(state.num_frames), int(loaded_e), int(back_cursor) + 1)
                if chunk_end <= chunk_start:
                    back_cursor -= max(1, int(getattr(state, "session_chunk_frames", chunk_size)))
                    continue

                local_prompt_frames = _chunk_prompt_frames(chunk_start, chunk_end)
                has_local_prompt = len(local_prompt_frames) > 0

                if has_local_prompt:
                    # If prompt exists in this chunk, sweep around that prompt.
                    start_rev = int(max(local_prompt_frames))
                    for item in _consume_chunk(chunk_start, chunk_end, start_idx=start_rev, reverse=True):
                        yield item
                    for item in _consume_chunk(chunk_start, chunk_end, start_idx=start_rev, reverse=False):
                        yield item
                elif carry_seed_boxes_back:
                    next_known = int(chunk_end) if int(chunk_end) < int(state.num_frames) else int(chunk_end - 1)
                    _inject_seed_boxes(
                        state,
                        processor,
                        frame_idx=int(chunk_end - 1),
                        seed_boxes=carry_seed_boxes_back,
                        original_size=orig_size,
                        seed_frame_idx=next_known,
                        mask_h=int(h),
                        mask_w=int(w),
                    )
                    for item in _consume_chunk(chunk_start, chunk_end, start_idx=int(chunk_end - 1), reverse=True):
                        yield item
                else:
                    near_seed_boxes, near_seed_frame = _find_seed_near_chunk(chunk_start, chunk_end)
                    if near_seed_boxes:
                        _inject_seed_boxes(
                            state,
                            processor,
                            frame_idx=int(chunk_end - 1),
                            seed_boxes=near_seed_boxes,
                            original_size=orig_size,
                            seed_frame_idx=near_seed_frame,
                            mask_h=int(h),
                            mask_w=int(w),
                        )
                        for item in _consume_chunk(chunk_start, chunk_end, start_idx=int(chunk_end - 1), reverse=True):
                            yield item
                    else:
                        for item in _mark_chunk_empty(chunk_start, chunk_end):
                            yield item

                # Safety net: if model iterator skipped frames, mark uncovered frames as empty.
                for item in _mark_chunk_empty(chunk_start, chunk_end):
                    yield item

                next_seed_back = _seed_boxes_from_chunk_scan(
                    state,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    h=int(h),
                    w=int(w),
                    prefer_end=False,
                )
                if next_seed_back:
                    carry_seed_boxes_back = next_seed_back

                back_cursor = int(chunk_start) - 1
    except Exception as e:
        if _JOB_USE_TORCH_COMPILE and _is_torch_compile_runtime_failure(e):
            _disable_torch_compile_runtime(str(e))
            for item in propagate_masks(state):
                yield item
            return
        raise
    finally:
        state.session_chunk_frames = int(orig_session_chunk_frames)

    # Final safety net for any residual unvisited frames.
    if len(visited_frames) < total:
        for item in _mark_chunk_empty(0, int(state.num_frames)):
            yield item

    covered = len(visited_frames)
    masked = len(masked_frames)
    state.current_frame_idx = int(np.clip(last_frame_idx, 0, max(0, state.num_frames - 1)))
    done_msg = f"Propagation done. masked={masked}/{total}, visited={covered}/{total}"
    write_status(state.job_dir, state="running", phase="propagate", progress=0.60, message=done_msg)
    yield (
        state,
        done_msg,
        gr.update(value=state.current_frame_idx),
        update_frame_display(state, state.current_frame_idx),
    )

# -------------------------
# Render GB/BB video and finish
# -------------------------

def _bg_bgr_from_mode(mode: str) -> Tuple[int,int,int]:
    m = (mode or "GB").upper()
    if m.startswith("GB"):
        return (0, 255, 0)  # BGR green
    return (255, 0, 0)      # BGR blue

def _set_event_later(ev: threading.Event, delay_sec: float = 1.0) -> None:
    def _w():
        time.sleep(delay_sec)
        ev.set()
    threading.Thread(target=_w, daemon=True).start()
    
def render_fg_black_video(
    state: AppState,
    mux_audio: bool = True,
    codec: str = None,
) -> str:
    """
    fg(鬲・ｺｷ・ｮ螢ｹﾎ樣垓雜｣・ｽ・ｯ) 驛｢・ｧ髮区ｩｸ・ｽ・ｿ郢晢ｽｻ隨倥・ﾂ蠅難ｽｻ阮吶・驍ｵ・ｺ陷ｷ・ｶ繝ｻ荵昴・鬩帙・・ｽ・ｦ遶擾ｽｽ繝ｻ・ｻ繝ｻ・ｶ2郢晢ｽｻ郢晢ｽｻ
    - union mask 髯ｷﾂ郢晢ｽｻ隨・ｽ｡驍ｵ・ｺ陞滂ｽｧ郢晢ｽｻRGB驍ｵ・ｲ遶乗劼・ｽ・､隰費ｽｶ郢晢ｽｻ鬲・ｺ倥・
    - 鬯ｮ・ｻ繝ｻ・ｳ髯橸ｽ｢繝ｻ・ｰ驍ｵ・ｺ繝ｻ・ｯ mux_audio 驍ｵ・ｺ郢晢ｽｻTrue 驍ｵ・ｺ繝ｻ・ｮ髯懶ｽ｣繝ｻ・ｴ髯ｷ・ｷ陋ｹ・ｻ郢晢ｽｻ驍ｵ・ｺ繝ｻ・ｿ髯ｷ蛹ｻ繝ｻ髯悟､青蛹・ｽｽ・ｻ髯具ｽｹ繝ｻ・ｺ鬯ｮ・｢髦ｮ蜊搾ｽｰ驛｢・ｧ隰梧汚・ｽ・ｻ陋滂ｽ･繝ｻ・ｰ驛｢・ｧ郢晢ｽｻ
    """
    if state is None or state.num_frames == 0:
        raise gr.Error("No frames loaded.")
    if not state.masks_by_frame:
        raise gr.Error("No masks found. Click/box prompt and/or propagate first.")

    out_dir = state.output_dir or state.job_dir
    os.makedirs(out_dir, exist_ok=True)

    s_ms = int(round(state.segment_start_sec * 1000.0))
    e_ms = int(round(state.segment_end_sec * 1000.0))
    seg_tag = f"{s_ms:09d}_{e_ms:09d}"
    out_noaudio = os.path.join(out_dir, f"sam3_fg_black_noaudio_{state.job_id}_{seg_tag}.mp4")
    out_final   = os.path.join(out_dir, f"sam3_fg_black_{state.job_id}_{seg_tag}.mp4")

    fps = float(state.video_fps) if state.video_fps and state.video_fps > 0 else 30.0
    first = state.video_frames[0]
    w, h = first.size[0], first.size[1]

    c = (codec or _JOB_GBBB_CODEC or "mp4v")
    c4 = (c + "mp4v")[:4]
    fourcc = cv2.VideoWriter_fourcc(*c4)
    writer = cv2.VideoWriter(out_noaudio, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open (codec={c4}).")

    write_status(state.job_dir, state="running", phase="render_fg", progress=0.62, message="Rendering fg (black bg)...")

    for idx in range(state.num_frames):
        frame_rgb = np.array(state.video_frames[idx], dtype=np.uint8)
        if frame_rgb.shape[0] != h or frame_rgb.shape[1] != w:
            frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        union = _union_mask_for_frame(state, idx, h, w)  # bool
        out_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        out_bgr[union] = frame_rgb[:, :, ::-1][union]
        writer.write(out_bgr)

        if (idx + 1) % 60 == 0 or (idx + 1) == state.num_frames:
            prog = 0.62 + 0.10 * ((idx + 1) / state.num_frames)
            write_status(state.job_dir, state="running", phase="render_fg", progress=prog,
                         message=f"Rendering fg... {idx+1}/{state.num_frames}")

    writer.release()

    if mux_audio:
        try:
            write_status(state.job_dir, state="running", phase="audio_fg", progress=0.73, message="Muxing audio (fg)...")
            mux_audio_from_original_segment(
                video_noaudio=out_noaudio,
                original_video=state.source_video_path,
                out_path=out_final,
                start_sec=state.segment_start_sec,
                end_sec=state.segment_end_sec,
            )
            return out_final
        except Exception as e:
            print("[WARN] fg audio mux failed:", e)
            return out_noaudio
    return out_noaudio
def render_mask_video(state: AppState, codec: str = None) -> str:
    """
    mask髯ｷ閧ｴ繝ｻ陋ｻ・､驛｢・ｧ髮区ｩｸ・ｽ・ｿ郢晢ｽｻ隨倥・ﾂ蠅難ｽｻ阮吶・驍ｵ・ｺ陷ｷ・ｶ繝ｻ荵昴・鬩帙・・ｽ・ｦ遶擾ｽｽ繝ｻ・ｻ繝ｻ・ｶ2郢晢ｽｻ郢晢ｽｻ
    - 0/255 驍ｵ・ｺ繝ｻ・ｮ 3ch mp4 驍ｵ・ｺ繝ｻ・ｨ驍ｵ・ｺ陷会ｽｱ遯ｶ・ｻ髣厄ｽｫ隴取得・ｽ・ｭ陋帙・・ｽ・ｼ闔蛹・ｽｽ・ｺ陷ｻ逎ｯ蜈ｱ髯ｷ繝ｻ・ｽ・ｪ髯ｷ驛∬か繝ｻ・ｼ郢晢ｽｻ
    """
    if state is None or state.num_frames == 0:
        raise gr.Error("No frames loaded.")
    if not state.masks_by_frame:
        raise gr.Error("No masks found. Click/box prompt and/or propagate first.")

    out_dir = state.output_dir or state.job_dir
    os.makedirs(out_dir, exist_ok=True)

    s_ms = int(round(state.segment_start_sec * 1000.0))
    e_ms = int(round(state.segment_end_sec * 1000.0))
    seg_tag = f"{s_ms:09d}_{e_ms:09d}"
    out_path = os.path.join(out_dir, f"sam3_mask_{state.job_id}_{seg_tag}.mp4")
    fps = float(state.video_fps) if state.video_fps and state.video_fps > 0 else 30.0
    first = state.video_frames[0]
    w, h = first.size[0], first.size[1]

    c = (codec or _JOB_GBBB_CODEC or "mp4v")
    c4 = (c + "mp4v")[:4]
    fourcc = cv2.VideoWriter_fourcc(*c4)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open (codec={c4}).")

    write_status(state.job_dir, state="running", phase="render_mask", progress=0.74, message="Rendering mask video...")
    for idx in range(state.num_frames):
        union = _union_mask_for_frame(state, idx, h, w)  # bool
        m = (union.astype(np.uint8) * 255)
        out_bgr = np.stack([m, m, m], axis=-1)  # 3ch
        writer.write(out_bgr)
        if (idx + 1) % 60 == 0 or (idx + 1) == state.num_frames:
            prog = 0.74 + 0.08 * ((idx + 1) / state.num_frames)
            write_status(state.job_dir, state="running", phase="render_mask", progress=prog,
                         message=f"Rendering mask... {idx+1}/{state.num_frames}")
    writer.release()
    return out_path


def render_gbbb_video(
    state: AppState,
    bg_mode: str,
    mux_audio: bool = True,
    codec: str = None,
) -> str:
    if state is None or state.num_frames == 0:
        raise gr.Error("No frames loaded.")
    if not state.masks_by_frame:
        raise gr.Error("No masks found. Click/box prompt and/or propagate first.")

    out_dir = state.output_dir or state.job_dir
    os.makedirs(out_dir, exist_ok=True)

    # filename safe tag (ms)
    s_ms = int(round(state.segment_start_sec * 1000.0))
    e_ms = int(round(state.segment_end_sec * 1000.0))
    seg_tag = f"{s_ms:09d}_{e_ms:09d}"
    bg = (bg_mode or "GB").upper()
    out_noaudio = os.path.join(out_dir, f"sam3_{bg}_noaudio_{state.job_id}_{seg_tag}.mp4")
    out_final = os.path.join(out_dir, f"sam3_{bg}_{state.job_id}_{seg_tag}.mp4")

    fps = float(state.video_fps) if state.video_fps and state.video_fps > 0 else 30.0
    first = state.video_frames[0]
    w, h = first.size[0], first.size[1]
    bg_bgr = _bg_bgr_from_mode(bg)

    write_status(state.job_dir, state="running", phase="render", progress=0.65, message="Rendering GB/BB video...")

    c = (codec or _JOB_GBBB_CODEC or "mp4v")
    c4 = (c + "mp4v")[:4]  # safety: ensure 4 chars
    fourcc = cv2.VideoWriter_fourcc(*c4)
    writer = cv2.VideoWriter(out_noaudio, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open (codec={c4}).")

    for idx in range(state.num_frames):
        frame_rgb = np.array(state.video_frames[idx], dtype=np.uint8)  # (h,w,3) RGB
        if frame_rgb.shape[0] != h or frame_rgb.shape[1] != w:
            frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        union = _union_mask_for_frame(state, idx, h, w)  # bool

        out_bgr = np.empty((h, w, 3), dtype=np.uint8)
        out_bgr[:] = bg_bgr
        out_bgr[union] = frame_rgb[:, :, ::-1][union]  # RGB->BGR

        writer.write(out_bgr)

        if (idx + 1) % 60 == 0 or (idx + 1) == state.num_frames:
            prog = 0.65 + 0.25 * ((idx + 1) / state.num_frames)
            write_status(state.job_dir, state="running", phase="render", progress=prog,
                         message=f"Rendering... {idx+1}/{state.num_frames}")

    writer.release()

    # audio mux
    if mux_audio:
        try:
            write_status(state.job_dir, state="running", phase="audio", progress=0.93, message="Muxing audio...")
            mux_audio_from_original_segment(
                video_noaudio=out_noaudio,
                original_video=state.source_video_path,
                out_path=out_final,
                start_sec=state.segment_start_sec,
                end_sec=state.segment_end_sec,
            )
        except Exception as e:
            # audio髯樊ｻゑｽｽ・ｱ髫ｰ・ｨ陷会ｽｱ邵ｲ蝣､・ｹ・ｧ郢ｧ莠包ｽｸ蜊諱崎叉蟯ｩ蜻ｳ驍ｵ・ｺ闔会ｽ｣郢晢ｽｻ髫ｹ・ｿ闕ｵ譏ｶ繝ｻ郢晢ｽｻ騾趣ｽｯ・ゑｽｰ鬨ｾ蛹・ｽｽ・ｨ髣包ｽｳ鬮ｮ竏晏牽驍ｵ・ｺ闕ｵ譎｢・ｽ荵昴・郢晢ｽｻ
            print("[WARN] audio mux failed:", e)
            out_final = out_noaudio
    else:
        out_final = out_noaudio

    write_status(state.job_dir, state="running", phase="render", progress=0.97, message="Render done.")
    return out_final


def render_preview_mp4(
    state: AppState,
    bg_mode: str,
    include_audio: bool = True,
    max_width: int = 960,
    crf: int = 23,
    preset: str = "veryfast",
    preview_fps: Optional[int] = None,
    audio_bitrate: str = "96k",
) -> str:
    """
    Render Preview 驍ｵ・ｺ繝ｻ・ｮ髮趣ｽｬ遶丞､ｲ・ｽ繝ｻ
      1) 髣包ｽｳ繝ｻ・ｭ鬯ｮ・｢郢晢ｽｻ cv2(VideoWriter)驍ｵ・ｺ繝ｻ・ｧGB/BB(noaudio)驛｢・ｧ陷代・・ｽ・ｽ隲帛･・ｽｽ繝ｻ
      2) 髯樊ｺｽ蛻､鬩ｪ・､: ffmpeg驍ｵ・ｺ繝ｻ・ｧ H.264(avc1) (+AAC) 驍ｵ・ｺ繝ｻ・ｮ 驕ｯ・ｶ隲帷ｿｫ繝ｻ驛｢譎｢・ｽ・ｬ驛｢譎∽ｾｭ・守､ｼ・ｹ譎｢・ｽ・ｼ鬨ｾ蛹・ｽｽ・ｨmp4驕ｯ・ｶ郢晢ｽｻ驍ｵ・ｺ繝ｻ・ｫ驍ｵ・ｺ陷ｷ・ｶ繝ｻ繝ｻ
    """
    if state is None or state.num_frames == 0:
        raise gr.Error("No frames loaded.")

    # 1) intermediate (GB/BB noaudio)
    write_status(state.job_dir, state="running", phase="preview_mid", progress=0.70, message="Rendering intermediate (no-audio)...")
    mid_noaudio = render_gbbb_video(state, bg_mode=bg_mode, mux_audio=False, codec=_JOB_GBBB_CODEC)

    # 2) preview path
    out_dir = state.output_dir or state.job_dir
    os.makedirs(out_dir, exist_ok=True)
    s_ms = int(round(state.segment_start_sec * 1000.0))
    e_ms = int(round(state.segment_end_sec * 1000.0))
    seg_tag = f"{s_ms:09d}_{e_ms:09d}"
    bg = (bg_mode or "GB").upper()
    preview_path = os.path.join(out_dir, f"sam3_{bg}_preview_{state.job_id}_{seg_tag}.mp4")

    write_status(state.job_dir, state="running", phase="preview_encode", progress=0.86, message="Encoding preview (H.264/AAC)...")
    if include_audio:
        make_h264_aac_preview(
            video_noaudio=mid_noaudio,
            original_video=state.source_video_path,
            out_path=preview_path,
            start_sec=state.segment_start_sec,
            end_sec=state.segment_end_sec,
            max_width=max_width,
            crf=crf,
            preset=preset,
            fps=preview_fps,
            audio_bitrate=audio_bitrate,
        )
    else:
        make_h264_preview_video_only(
            video_noaudio=mid_noaudio,
            out_path=preview_path,
            max_width=max_width,
            crf=crf,
            preset=preset,
            fps=preview_fps,
        )

    state.preview_mp4_path = preview_path
    write_status(state.job_dir, state="running", phase="preview_done", progress=0.92, message="Preview ready.")
    return preview_path


def finish_job(
    state: AppState,
    mux_audio: bool,
    done_event: threading.Event,
    preview_bg_mode: str = "",
) -> str:
    """
    Render -> write result.json -> write status -> (optional) close signal
    Returns: message only (do NOT return video path to Gradio)
    """
    try:
        requested_insert_mode = str(_JOB_INSERT_MODE_DEFAULT or "transparent").strip()
        requested_insert_up = requested_insert_mode.upper()

        default_bg = str(_JOB_BG_MODE_DEFAULT or "GB").upper()
        if default_bg not in ("GB", "BB"):
            default_bg = "GB"
        preview_bg_up = str(preview_bg_mode or "").upper()
        if preview_bg_up not in ("GB", "BB"):
            preview_bg_up = default_bg

        # Transparent import currently has runtime issues in AviUtl-side apply.
        # To keep behavior consistent with Render Preview, force GB/BB output.
        fallback_from_transparent = False
        if requested_insert_up in ("GB", "BB"):
            effective_insert_mode = requested_insert_up
        else:
            effective_insert_mode = preview_bg_up
            fallback_from_transparent = (requested_insert_up in ("TRANSPARENT", "ALPHA", "A"))

        composited_path = render_gbbb_video(
            state,
            bg_mode=effective_insert_mode,
            mux_audio=bool(mux_audio),
            codec=_JOB_GBBB_CODEC,
        )
        output_dir = state.output_dir or state.job_dir
        source_stem = _source_stem_for_output(state.source_video_path)
        named_final_path = os.path.join(output_dir, f"{source_stem}_{effective_insert_mode}.mp4")
        out_path = _safe_replace_file(composited_path, named_final_path)
        composited_path = out_path
        stats = {"num_frames": int(state.num_frames), "fps": float(state.video_fps)}
        insert_mode_for_result = effective_insert_mode
        bg_mode_used = effective_insert_mode

        note = ""
        if fallback_from_transparent:
            note = (
                f"Transparent mode fallback is active. "
                f"Used {effective_insert_mode} output to match Render Preview."
            )
            print("[WARN]", note)

        # Remove other files generated for this job (preview/intermediate/noaudio, etc.).
        _cleanup_job_output_artifacts(output_dir, state.job_id, keep_paths=[out_path])
        state.preview_mp4_path = ""

        write_result(
            state.job_dir,
            True,
            output_video_path=out_path,
            composited_video_path=composited_path,
            bg_mode=bg_mode_used,
            insert_mode=insert_mode_for_result,
            stats=stats,
            prompt_mode=_JOB_PROMPT_MODE,
            output_mode=_JOB_OUTPUT_MODE,
        )
        write_status(state.job_dir, state="done", phase="done", progress=1.0, message="Done. You can close this tab.")

        # Delay close signal slightly so the UI can reflect the done state.
        _set_event_later(done_event, delay_sec=1.0)

        msg = (
            "**Finish complete (result.json written).**\n\n"
            f"- Output: `{out_path}`\n"
            "- AviUtl2 will read `result.json` and import the clip.\n"
            "- You can close this tab.\n"
        )
        if note:
            msg += f"- {note}\n"
        return msg
    except Exception as e:
        import traceback
        traceback.print_exc()
        write_result(state.job_dir, False, error_message=str(e))
        write_status(state.job_dir, state="failed", phase="done", progress=1.0, message=f"Failed: {e}")

        _set_event_later(done_event, delay_sec=1.0)

        return f"**Failed**: {e}"

def fail_job(state: AppState, done_event: threading.Event) -> str:
    write_result(state.job_dir, False, error_message="User pressed Fail.")
    write_status(state.job_dir, state="failed", phase="done", progress=1.0, message="Failed by user.")
    done_event.set()
    return "Marked failed. You can close this tab."

def main():
    global _STATUS_WRITE_CONSECUTIVE_FAILS, _STATUS_WRITE_LAST_ERROR
    _STATUS_WRITE_CONSECUTIVE_FAILS = 0
    _STATUS_WRITE_LAST_ERROR = ""

    ap = argparse.ArgumentParser()
    ap.add_argument("--job_dir", required=True)
    args = ap.parse_args()

    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(job_dir)
    setup_proxy_safety()

    print("=== sam3_gradio_job start ===")
    print("job_dir =", str(job_dir))
    print("HTTP_PROXY =", os.environ.get("HTTP_PROXY"))
    print("NO_PROXY =", os.environ.get("NO_PROXY"))

    req_path = job_dir / "request.json"
    if not req_path.exists():
        write_result(str(job_dir), False, error_message="request.json not found")
        write_status(str(job_dir), state="failed", phase="boot", progress=1.0, message="request.json not found")
        return

    try:
        with open(req_path, "r", encoding="utf-8-sig") as f:
            req = json.load(f)
    except Exception as e:
        traceback.print_exc()
        write_result(str(job_dir), False, error_message=f"request.json read failed: {e}")
        write_status(str(job_dir), state="failed", phase="boot", progress=1.0, message=f"request.json read failed: {e}")
        return

    try:
        _apply_request_defaults(req)
    except Exception as e:
        traceback.print_exc()
        write_result(str(job_dir), False, error_message=f"request defaults apply failed: {e}")
        write_status(str(job_dir), state="failed", phase="boot", progress=1.0, message=f"request defaults apply failed: {e}")
        return
    try:
        _ensure_cuda_triton_ready()
    except Exception as e:
        write_result(str(job_dir), False, error_message=str(e))
        write_status(str(job_dir), state="failed", phase="boot", progress=1.0, message=str(e))
        return
    try:
        _ensure_pyav_ready()
    except Exception as e:
        write_result(str(job_dir), False, error_message=str(e))
        write_status(str(job_dir), state="failed", phase="boot", progress=1.0, message=str(e))
        return

    print("DEVICE =", _JOB_DEVICE, "DTYPE =", str(_JOB_DTYPE))
    print(
        "SESSION_DEVICES =",
        f"processing={_JOB_SESSION_PROCESSING_DEVICE},",
        f"video_storage={_JOB_SESSION_VIDEO_STORAGE_DEVICE},",
        f"state={_JOB_SESSION_INFERENCE_STATE_DEVICE}",
    )
    print("INSERT_MODE =", _JOB_INSERT_MODE_DEFAULT)

    job_id = req.get("job_id", "unknown")
    src = req.get("source_video_path", "")
    start = float(req.get("playback_start_sec", 0.0) or 0.0)
    end = float(req.get("playback_end_sec", 0.0) or 0.0)
    out_dir = req.get("output_dir", str(job_dir)) or str(job_dir)
    timeline = req.get("timeline", {}) or {}
    try:
        timeline_start_frame = int(timeline.get("start_frame", -1))
    except Exception:
        timeline_start_frame = -1
    try:
        timeline_end_frame = int(timeline.get("end_frame", -1))
    except Exception:
        timeline_end_frame = -1
    try:
        timeline_fps = float(timeline.get("fps", 0.0) or 0.0)
    except Exception:
        timeline_fps = 0.0
    timeline_num_frames = 0
    if timeline_start_frame >= 0 and timeline_end_frame >= timeline_start_frame:
        timeline_num_frames = timeline_end_frame - timeline_start_frame + 1
    timeline_duration_sec = 0.0
    if timeline_num_frames > 0 and timeline_fps > 0.0:
        timeline_duration_sec = timeline_num_frames / timeline_fps
        print(
            "TIMELINE_SEGMENT =",
            f"frames={timeline_num_frames}, fps={timeline_fps:.6f}, duration={timeline_duration_sec:.6f}s",
        )

    options = req.get("options", {}) or {}
    # 鬯ｮ貊ゑｽｽ・ｷ髯昴・・ｽ・ｺ驍ｵ・ｺ繝ｻ・ｧ驍ｵ・ｺ繝ｻ・ｮ驛｢譎｢・ｽ・｡驛｢譎｢・ｽ・｢驛｢譎｢・ｽ・ｪ髫ｴ・ｫ繝ｻ・ｯ髮九ｅ繝ｻ繝ｻ蟶晢ｽｩ蛹・ｽｽ・ｿ驍ｵ・ｺ闔会ｽ｣繝ｻ迢暦ｽｸ・ｺ雋・∞・ｽ竏ｫ・ｸ・ｲ遶擾ｽｫ隨冗霜蟠輔・・ｶ鬯ｮ・ｯ陷亥沺・ｬ・ｰ髯橸ｽｳ郢晢ｽｻ<=0)驍ｵ・ｺ繝ｻ・ｯ髣包ｽｳ闔ｨ竏晏ｿ憺し・ｺ繝ｻ・ｫ髣包ｽｳ繝ｻ・ｸ驛｢・ｧ遶丞､ｲ・ｽ繝ｻ
    ms_val = options.get("max_seconds", 600)
    try:
        max_seconds = 600 if ms_val is None else int(ms_val)
    except Exception:
        max_seconds = 600
    if max_seconds <= 0:
        max_seconds = 600

    # Keep timeline duration as the primary source of truth for frame-accurate
    # alignment with the editor. Playback range is used as fallback.
    segment_start_sec = float(start)
    playback_duration_sec = float(end - start)
    if timeline_duration_sec > 0.0:
        segment_duration_sec = float(timeline_duration_sec)
    elif playback_duration_sec > 0.0:
        segment_duration_sec = float(playback_duration_sec)
    else:
        write_result(str(job_dir), False, error_message=f"Invalid segment duration: {playback_duration_sec}")
        write_status(
            str(job_dir),
            state="failed",
            phase="load_video",
            progress=1.0,
            message=f"Invalid segment duration: {playback_duration_sec}",
        )
        return

    if timeline_duration_sec > 0.0 and playback_duration_sec > 0.0:
        tol = max(0.05, 1.5 / max(1.0, timeline_fps))
        delta = abs(float(playback_duration_sec) - float(timeline_duration_sec))
        if delta > tol:
            print(
                "[WARN] playback/timeline duration mismatch:",
                f"playback={playback_duration_sec:.6f}s,",
                f"timeline={timeline_duration_sec:.6f}s,",
                "using timeline range",
            )

    if segment_duration_sec > float(max_seconds):
        print(
            "[WARN] segment duration clipped by max_seconds:",
            f"duration={segment_duration_sec:.6f}s -> {float(max_seconds):.6f}s",
        )
        segment_duration_sec = float(max_seconds)

    segment_end_sec = segment_start_sec + segment_duration_sec

    expected_num_frames = 0
    if timeline_num_frames > 0 and timeline_fps > 0.0:
        tol = max(0.05, 1.5 / max(1.0, timeline_fps))
        if abs(timeline_duration_sec - segment_duration_sec) <= tol:
            expected_num_frames = int(timeline_num_frames)
        else:
            expected_num_frames = max(1, int(round(segment_duration_sec * timeline_fps)))

    print(
        "SEGMENT_REQUEST =",
        f"start={segment_start_sec:.6f}, end={segment_end_sec:.6f}, duration={segment_duration_sec:.6f}s,",
        f"expected_frames={expected_num_frames}",
    )

    # status: boot
    write_status(str(job_dir), state="running", phase="boot", progress=0.01, message="Loading model...")

    # load model
    try:
        ensure_models_loaded()
    except Exception as e:
        traceback.print_exc()
        write_result(str(job_dir), False, error_message=f"Model load failed: {e}")
        write_status(str(job_dir), state="failed", phase="boot", progress=1.0, message=f"Model load failed: {e}")
        return

    write_status(
        str(job_dir),
        state="running",
        phase="preprocess",
        progress=0.06,
        message="Preprocessing segment video (ffmpeg)...",
    )

    if not src:
        write_result(str(job_dir), False, error_message="source_video_path is empty")
        write_status(str(job_dir), state="failed", phase="load_video", progress=1.0, message="source_video_path is empty")
        return

    preprocessed_input_video = str(job_dir / "_segment_input_preprocessed.mp4")
    preprocessed_created = False
    try:
        preprocess_video_segment_for_tracking(
            source_video=src,
            out_path=preprocessed_input_video,
            start_sec=segment_start_sec,
            duration_sec=segment_duration_sec,
            fps=(timeline_fps if timeline_fps > 0.0 else None),
            expected_num_frames=(expected_num_frames if expected_num_frames > 0 else None),
        )
        preprocessed_created = True
    except Exception as e:
        traceback.print_exc()
        write_result(str(job_dir), False, error_message=f"Segment preprocess failed: {e}")
        write_status(str(job_dir), state="failed", phase="preprocess", progress=1.0, message=f"Segment preprocess failed: {e}")
        return

    write_status(str(job_dir), state="running", phase="load_video", progress=0.08, message="Loading preprocessed segment metadata...")

    # load segment frames
    try:
        preview_frames, seginfo = load_video_segment_frames(
            preprocessed_input_video,
            0.0,
            segment_duration_sec,
            max_seconds=0,
            expected_num_frames=(expected_num_frames if expected_num_frames > 0 else None),
            load_all=False,
        )
    except Exception as e:
        traceback.print_exc()
        if preprocessed_created:
            _safe_remove_file(preprocessed_input_video)
        write_result(str(job_dir), False, error_message=f"Video load failed: {e}")
        write_status(str(job_dir), state="failed", phase="load_video", progress=1.0, message=f"Video load failed: {e}")
        return

    write_status(str(job_dir), state="running", phase="init_session", progress=0.18, message="Initializing SAM3 video session...")

    # init inference session
    state0 = AppState()
    state0.job_id = str(job_id)
    state0.job_dir = str(job_dir)
    state0.output_dir = str(out_dir)
    state0.source_video_path = str(src)
    state0.segment_start_sec = float(segment_start_sec)
    effective_end_sec = float(segment_start_sec + (float(seginfo.num_frames) / max(1e-6, float(seginfo.fps))))
    state0.segment_end_sec = float(effective_end_sec)
    if abs(float(effective_end_sec) - float(segment_end_sec)) > (0.5 / max(1e-6, float(seginfo.fps))):
        print(
            "[INFO] Segment end adjusted for frame-accurate sync:",
            f"requested_end={float(segment_end_sec):.6f}s -> effective_end={float(effective_end_sec):.6f}s",
            f"(frames={int(seginfo.num_frames)}, fps={float(seginfo.fps):.6f})",
        )
    state0.segment_video_path = str(preprocessed_input_video)
    state0.segment_video_owned = bool(preprocessed_created)
    state0.video_frames = OnDemandVideoFrames(
        video_path=preprocessed_input_video,
        num_frames=seginfo.num_frames,
        fps=seginfo.fps,
        width=seginfo.width,
        height=seginfo.height,
        cache_max=_JOB_ONDEMAND_CACHE_MAX_FRAMES,
    )
    state0.video_fps = float(seginfo.fps)
    state0.session_chunk_frames = max(32, int(_JOB_SESSION_CHUNK_FRAMES))
    _fw, _fh = preview_frames[0].size
    state0.original_frame_size = (int(_fh), int(_fw))

    # Initialize first streaming chunk session (avoid full-video allocation).
    try:
        profiles = _session_device_profiles()
        profile_idx = 0
        while True:
            pdev, vdev, sdev = profiles[min(profile_idx, len(profiles) - 1)]
            _set_session_devices(pdev, vdev, sdev)
            try:
                ensure_inference_session_for_frame(state0, 0, force_reload=True)
                break
            except Exception as e:
                msg = str(e)
                low_mem = _is_low_memory_error(e)
                if low_mem and profile_idx + 1 < len(profiles):
                    profile_idx += 1
                    npdev, nvdev, nsdev = profiles[profile_idx]
                    write_status(
                        str(job_dir),
                        state="running",
                        phase="init_session",
                        progress=0.18,
                        message=(
                            "Memory pressure detected. Retrying session init with devices "
                            f"processing={npdev}, video_storage={nvdev}, state={nsdev}..."
                        ),
                    )
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    gc.collect()
                    continue
                raise

        # 髣包ｽｳ鬯ｩ蟷｢・ｽ・ｨ驛｢譎√・郢晢ｽｻ驛｢・ｧ繝ｻ・ｸ驛｢譎｢・ｽ・ｧ驛｢譎｢・ｽ・ｳ驍ｵ・ｺ繝ｻ・ｧ髯橸ｽｻ隶灘･・ｽｽ・ｧ驍ｵ・ｺ隰疲ｻゑｽｽ・ｿ郢晢ｽｻ繝ｻ・ｦ遶丞｣ｺ繝ｻ驍ｵ・ｺ髦ｮ蜷ｮ繝ｻ驍ｵ・ｺ陟募ｨｯ譌ｺ驛｢・ｧ郢晢ｽｻ
        gc.collect()
    except Exception as e:
        traceback.print_exc()
        _release_app_state(state0)
        write_result(str(job_dir), False, error_message=f"init_video_session failed: {e}")
        write_status(str(job_dir), state="failed", phase="init_session", progress=1.0, message=f"init_video_session failed: {e}")
        return

    write_status(str(job_dir), state="running", phase="ui", progress=0.25, message="Launching Gradio...")

    done_event = threading.Event()
    job_key = str(job_id)
    _register_app_state(job_key, state0)

    # Build UI
    with gr.Blocks(title=f"SAM3 Job {job_id}") as demo:
        app_state = gr.State(job_key)
        gr.Markdown("## SAM3 BB/GB Generator")
        # gr.Markdown(f"- job_id: `{job_id}`")
        # gr.Markdown(f"- source_video_path: `{src}`")
        # gr.Markdown(f"- segment: `{seginfo.start_sec:.3f}` sec 驕ｶ鄙ｫ繝ｻ`{seginfo.end_sec:.3f}` sec")
        gr.Markdown(
            f"- loaded frames: `{seginfo.num_frames}` @ `{seginfo.fps:.3f}` fps "
            f"(device: `{_JOB_DEVICE}`, dtype: `{_JOB_DTYPE}`, chunk: `{state0.session_chunk_frames}`)"
        )

        with gr.Row():
            preview = gr.Image(label="Preview (click to add prompts)", value=preview_frames[0], interactive=False)
            with gr.Column(scale=0, min_width=240):
                frame_slider = gr.Slider(label="Frame", minimum=0, maximum=max(0, seginfo.num_frames - 1), step=1, value=0)
                propagate_btn = gr.Button("Propagate across segment", variant="primary")
                reset_prompts_btn = gr.Button("Reset prompts (keep video)", variant="secondary")
                propagate_status = gr.Markdown(value="")

        with gr.Row():
            obj_id_inp = gr.Number(value=1, precision=0, label="Object ID")
            label_radio = gr.Radio(choices=["positive", "negative"], value="positive", label="Point label")
            clear_old_chk = gr.Checkbox(value=False, label="Clear clicks for this Object ID (this frame)")
            prompt_type = gr.Radio(choices=["Points", "Boxes"], value="Points", label="Prompt type")

        with gr.Row():
            bg_choice = gr.Radio(choices=["GB", "BB"], value=_JOB_BG_MODE_DEFAULT, label="Preview BG (GB/BB)")
            mux_audio_chk = gr.Checkbox(value=_JOB_AUDIO_MUX_DEFAULT, label="Mux audio (ffmpeg)")
        preview_video = gr.Video(label="Render Preview (H.264/AAC)", interactive=False)
        with gr.Row():
            # Render Preview: 驕ｯ・ｶ隲帑ｼ夲ｽｽ・ｸ繝ｻ・ｭ鬯ｮ・｢陷・ｽｪ郢晢ｽｻpreview髯樊ｺｽ蛻､鬩ｪ・､驕ｯ・ｶ郢晢ｽｻ驍ｵ・ｺ繝ｻ・ｧ H.264/AAC mp4 驛｢・ｧ陷代・・ｽ・ｽ隲帛ｲｩ螟｢驍ｵ・ｺ繝ｻ・ｦ Gradio 驍ｵ・ｺ繝ｻ・ｫ鬮ｯ・ｦ繝ｻ・ｨ鬩穂ｼ夲ｽｽ・ｺ
            render_preview_btn = gr.Button("Render Preview (H.264/AAC)", variant="secondary")
            finish_btn = gr.Button("Finish (write result.json / no preview)", variant="primary")
            fail_btn = gr.Button("Fail", variant="secondary")

        out_msg = gr.Markdown(value="")

        # callbacks
        def _require_state(job_key_in: str) -> AppState:
            state_in = _get_app_state(job_key_in)
            if state_in is None:
                raise gr.Error("Session state was released. Please restart this job.")
            return state_in

        def _sync_frame(job_key_in: str, idx: int):
            state_in = _get_app_state(job_key_in)
            if state_in is None or state_in.num_frames <= 0:
                return gr.update()
            state_in.current_frame_idx = int(np.clip(int(idx), 0, state_in.num_frames - 1))
            return update_frame_display(state_in, state_in.current_frame_idx)

        frame_slider.change(fn=_sync_frame, inputs=[app_state, frame_slider], outputs=preview)

        def _click(
            img,
            job_key_in: str,
            frame_idx: int,
            obj_id: int,
            label: str,
            clear_old: bool,
            ptype: str,
            evt: gr.SelectData,
        ):
            state_in = _require_state(job_key_in)
            state_in.current_frame_idx = int(frame_idx)
            state_in.current_obj_id = int(obj_id)
            state_in.current_label = str(label)
            state_in.current_clear_old = bool(clear_old)
            state_in.current_prompt_type = str(ptype)
            img_out, _ = on_image_click(img, state_in, frame_idx, obj_id, label, clear_old, ptype, evt)
            return img_out

        preview.select(
            fn=_click,
            inputs=[preview, app_state, frame_slider, obj_id_inp, label_radio, clear_old_chk, prompt_type],
            outputs=[preview],
        )

        def _propagate(job_key_in: str):
            state_in = _get_app_state(job_key_in)
            if state_in is None:
                yield "Load failed.", gr.update(), gr.update()
                return
            try:
                for _state, status, slider_update, img in propagate_masks(state_in):
                    yield status, slider_update, img
            except Exception as e:
                traceback.print_exc()
                emsg = f"Propagate failed: {e}"
                if state_in.job_dir:
                    write_status(
                        state_in.job_dir,
                        state="failed",
                        phase="propagate",
                        progress=1.0,
                        message=emsg,
                    )
                cur = int(np.clip(int(state_in.current_frame_idx), 0, max(0, state_in.num_frames - 1)))
                yield emsg, gr.update(value=cur), update_frame_display(state_in, cur)

        propagate_btn.click(
            fn=_propagate,
            inputs=[app_state],
            outputs=[propagate_status, frame_slider, preview],
        )

        def _reset_prompts(job_key_in: str, frame_idx: int):
            state_in = _get_app_state(job_key_in)
            if state_in is None or state_in.num_frames == 0:
                return gr.update(), "No video loaded.", ""

            keep_idx = int(np.clip(int(frame_idx), 0, state_in.num_frames - 1))
            reset_prompts_keep_video(state_in)
            state_in.current_frame_idx = keep_idx

            img = update_frame_display(state_in, keep_idx)
            msg = "**Prompts cleared.** Clicks/boxes/masks were reset. Video frames were kept."
            return img, msg, ""

        reset_prompts_btn.click(
            fn=_reset_prompts,
            inputs=[app_state, frame_slider],
            outputs=[preview, out_msg, propagate_status],
        )

        def _render_preview(job_key_in: str, bg: str, include_audio: bool):
            state_in = _get_app_state(job_key_in)
            if state_in is None:
                return "Preview failed: session state missing.", gr.update(value=None)
            try:
                path = render_preview_mp4(
                    state_in,
                    bg_mode=bg,
                    include_audio=bool(include_audio),
                    max_width=1920,
                    crf=23,
                    preset="veryfast",
                    preview_fps=None,
                    audio_bitrate="96k",
                )
                if state_in.job_dir:
                    write_status(
                        state_in.job_dir,
                        state="waiting_user",
                        phase="ui",
                        progress=1.0,
                        message="Preview ready.",
                    )
                msg = (
                    "**Render Preview complete (H.264/AAC).**\n\n"
                    f"- `{path}`\n"
                    "- `result.json` is not written yet. It is written on Finish."
                )
                return msg, path
            except gr.Error as e:
                emsg = str(e).strip("'")
                if state_in.job_dir:
                    write_status(
                        state_in.job_dir,
                        state="waiting_user",
                        phase="ui",
                        progress=1.0,
                        message=f"Preview failed: {emsg}",
                    )
                return f"Preview failed: {emsg}", gr.update(value=None)
            except Exception as e:
                traceback.print_exc()
                emsg = str(e)
                if state_in.job_dir:
                    write_status(
                        state_in.job_dir,
                        state="waiting_user",
                        phase="ui",
                        progress=1.0,
                        message=f"Preview failed: {emsg}",
                    )
                return f"Preview failed: {emsg}", gr.update(value=None)

        render_preview_btn.click(
            fn=_render_preview,
            inputs=[app_state, bg_choice, mux_audio_chk],
            outputs=[out_msg, preview_video],
        )

        def _finish(job_key_in: str, mux_audio: bool, bg: str) -> str:
            state_in = _get_app_state(job_key_in)
            if state_in is None:
                return "Failed: session state missing. Please rerun."
            return finish_job(
                state_in,
                mux_audio=bool(mux_audio),
                done_event=done_event,
                preview_bg_mode=bg,
            )

        finish_btn.click(
            fn=_finish,
            inputs=[app_state, mux_audio_chk, bg_choice],
            outputs=[out_msg],
        )

        def _fail(job_key_in: str):
            state_in = _get_app_state(job_key_in)
            if state_in is None:
                return "Failed: session state missing."
            return fail_job(state_in, done_event=done_event)

        fail_btn.click(fn=_fail, inputs=[app_state], outputs=[out_msg])
    # launch server
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=None,
            share=False,
            prevent_thread_lock=True,
            quiet=False,
            inbrowser=False,
            allowed_paths=[str(Path(state0.output_dir).resolve())],
        )
        local_url = getattr(demo, "local_url", "") or ""
        print("local_url =", local_url)
        write_status(str(job_dir), state="waiting_user", phase="ui", progress=1.0, message="Gradio is running.", gradio_url=local_url)
    except Exception as e:
        traceback.print_exc()
        write_result(str(job_dir), False, error_message=str(e))
        write_status(str(job_dir), state="failed", phase="boot", progress=1.0, message=f"launch failed: {e}", gradio_url="")
        _unregister_app_state(job_key)
        return

    # keep alive until done
    wait_started = time.time()
    wait_timed_out = False
    while not done_event.is_set():
        if _JOB_UI_WAIT_TIMEOUT_SEC > 0:
            elapsed = time.time() - wait_started
            if elapsed >= float(_JOB_UI_WAIT_TIMEOUT_SEC):
                wait_timed_out = True
                break
        time.sleep(0.2)

    if wait_timed_out:
        msg = f"UI wait timeout: user did not finish within {_JOB_UI_WAIT_TIMEOUT_SEC} sec."
        print("[ERROR]", msg)
        write_result(str(job_dir), False, error_message=msg)
        write_status(str(job_dir), state="failed", phase="ui_timeout", progress=1.0, message=msg)
        done_event.set()

    try:
        demo.close()
    except Exception:
        pass

    _unregister_app_state(job_key)
    print("=== Done: exiting ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        # Last-resort failure reporting so the launcher can always consume result.json.
        try:
            job_dir_arg = ""
            argv = sys.argv[1:]
            for i, a in enumerate(argv):
                if a == "--job_dir" and i + 1 < len(argv):
                    job_dir_arg = argv[i + 1]
                    break
                if a.startswith("--job_dir="):
                    job_dir_arg = a.split("=", 1)[1]
                    break
            if job_dir_arg:
                Path(job_dir_arg).mkdir(parents=True, exist_ok=True)
                write_result(job_dir_arg, False, error_message=f"Unhandled exception: {e}")
                write_status(job_dir_arg, state="failed", phase="boot", progress=1.0, message=f"Unhandled exception: {e}")
        except Exception:
            pass
        raise

