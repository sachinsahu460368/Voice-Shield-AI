"""
VoiceShield-AI Backend — Phase 5 (Production Hardened)

This is the FastAPI server that receives audio files from the React
frontend, validates them, and returns analysis results.

Changes from Phase 3:
- AASIST-L deepfake detection model integrated (Phase 5)
- Model pre-loaded at startup to avoid first-request timeout
- PyTorch thread count limited for constrained environments (Render)
- Stage-by-stage logging for production debugging
"""

import logging
import os
import time
import uuid
import tempfile

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.audio_processor import preprocess_audio, check_ffmpeg
from services.deepfake_detector import detect_deepfake_longaudio, _load_model
from services.risk_engine import RiskEngine

# ---------- Logging ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("voiceshield.main")

# ---------- PyTorch thread limits ----------
# Render free-tier containers have limited CPU cores.  Without this cap
# PyTorch spawns one thread per visible CPU (often 4-8), causing memory
# pressure and context-switch overhead that leads to 502 timeouts.

_TORCH_THREADS = int(os.environ.get("TORCH_THREADS", "2"))
torch.set_num_threads(_TORCH_THREADS)
torch.set_num_interop_threads(1)
logger.info(
    "PyTorch threads: intra-op=%d  inter-op=%d",
    torch.get_num_threads(),
    torch.get_num_interop_threads(),
)

# ---------- App setup ----------

app = FastAPI(
    title="VoiceShield-AI",
    description="AI-Powered Voice Deepfake Detection API",
    version="0.1.0",
)

# ---------- CORS ----------
# Allow the Vite dev server origins AND any Render production URL.
# The RENDER_EXTERNAL_URL env var is set automatically on Render.

_cors_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
]

_render_url = os.environ.get("RENDER_EXTERNAL_URL")
if _render_url:
    _cors_origins.append(_render_url)

# Also accept a custom frontend URL if provided
_frontend_url = os.environ.get("FRONTEND_URL")
if _frontend_url:
    _cors_origins.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Startup: pre-load model ----------
# Loading the AASIST-L model lazily on the first /api/analyze request
# causes a 10-15 second delay that almost always exceeds Render's 30s
# request timeout.  Pre-loading at startup means the model is warm
# before any request arrives.

@app.on_event("startup")
def preload_model():
    """Pre-load the AASIST-L model during server startup."""
    logger.info("=== SERVER STARTUP: pre-loading AASIST-L model ===")
    t0 = time.time()
    try:
        model, device = _load_model()
        elapsed = time.time() - t0
        logger.info(
            "Model pre-loaded successfully on %s in %.2f s", device, elapsed,
        )
    except FileNotFoundError:
        logger.error(
            "AASIST-L checkpoint not found — /api/analyze will return 500 "
            "until the checkpoint is deployed."
        )
    except Exception:
        logger.exception("Failed to pre-load model — will retry on first request")

    # Also verify FFmpeg is available
    if check_ffmpeg():
        logger.info("FFmpeg is available")
    else:
        logger.warning(
            "FFmpeg not found at startup — audio preprocessing will fail"
        )

    logger.info("=== SERVER STARTUP COMPLETE ===")


# ---------- Allowed audio MIME types ----------
# We check the Content-Type header sent by the browser.
# This is not foolproof, but catches obvious mistakes.

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",           # .mp3
    "audio/wav",            # .wav
    "audio/x-wav",          # .wav (alternative)
    "audio/mp4",            # .m4a
    "audio/x-m4a",          # .m4a (alternative)
    "audio/ogg",            # .ogg
    "audio/flac",           # .flac
    "audio/webm",           # browser recordings
    "video/mp4",            # .mp4 (the frontend accepts video/* too)
    "video/webm",           # .webm video
}

# ---------- Allowed audio file extensions ----------
# Fallback for clients (e.g. curl) that send application/octet-stream.

ALLOWED_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4",
}

# Max file size: 50 MB
MAX_FILE_SIZE = 50 * 1024 * 1024


# ---------- Health check ----------

@app.get("/api/health")
def health_check():
    """Simple health check so we can verify the server is running."""
    return {
        "status": "ok",
        "service": "VoiceShield-AI backend",
        "version": "0.1.0",
    }


@app.get("/api/system-check")
def system_check():
    """Detailed system diagnostic for Render investigation."""
    import sys
    import platform
    from services.audio_processor import get_ffmpeg_executable

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_cuda": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "ffmpeg_available": os.path.exists(get_ffmpeg_executable()),
        "ffmpeg_path": get_ffmpeg_executable(),
    }


# ---------- Analyze endpoint ----------

@app.post("/api/analyze")
async def analyze_audio(audio: UploadFile = File(...)):
    """
    Receive an audio file, validate it, run (mock) analysis,
    and return the result.

    The frontend sends:  FormData with key "audio"
    We return:           { verdict, confidence, explanation }
    """
    request_id = uuid.uuid4().hex[:8]
    t_request = time.time()
    logger.info("[%s] === /api/analyze START === filename=%s", request_id, audio.filename)

    # 1. Check that a file was actually uploaded
    if not audio or not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    # 2. Check MIME type, with extension-based fallback for clients
    #    (e.g. curl) that send application/octet-stream.
    content_type = audio.content_type or ""
    file_ext = os.path.splitext(audio.filename)[1].lower()
    logger.info(
        "[%s] MIME check: content_type=%s  ext=%s",
        request_id, content_type, file_ext,
    )

    if content_type not in ALLOWED_AUDIO_TYPES:
        # Allow if the Content-Type is generic but the extension is valid
        if content_type == "application/octet-stream" and file_ext in ALLOWED_AUDIO_EXTENSIONS:
            logger.info(
                "[%s] Accepted via extension fallback (Content-Type: %s, ext: %s)",
                request_id, content_type, file_ext,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: Content-Type '{content_type}', "
                       f"extension '{file_ext}'. "
                       f"Please upload an audio file (MP3, WAV, M4A, etc.).",
            )

    # 3. Read file content and check size
    t_read = time.time()
    file_content = await audio.read()
    logger.info(
        "[%s] File read: %d bytes in %.2f ms",
        request_id, len(file_content), (time.time() - t_read) * 1000,
    )

    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is "
                   f"{MAX_FILE_SIZE // (1024 * 1024)} MB.",
        )

    # 4. Save to a temporary file (for future real processing)
    temp_path = None
    try:
        # Create a safe temp filename (never trust the original name)
        suffix = os.path.splitext(audio.filename)[1] or ".bin"
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=suffix,
            prefix=f"voiceshield_{uuid.uuid4().hex[:8]}_",
        )

        with os.fdopen(temp_fd, "wb") as f:
            f.write(file_content)

        logger.info("[%s] Temp file saved: %s", request_id, temp_path)

        # 5. ---- AUDIO PREPROCESSING ----
        logger.info("[%s] STAGE: Audio preprocessing START", request_id)
        t_preprocess = time.time()

        preprocessing = preprocess_audio(
            input_path=temp_path,
            original_filename=audio.filename,
        )

        preprocess_ms = (time.time() - t_preprocess) * 1000
        logger.info(
            "[%s] STAGE: Audio preprocessing END (%.0f ms, success=%s)",
            request_id, preprocess_ms, preprocessing.get("success"),
        )

        if not preprocessing.get("success"):
            error_msg = preprocessing.get(
                "error",
                "Audio preprocessing failed for an unknown reason.",
            )
            logger.warning(
                "[%s] Preprocessing failed: %s", request_id, error_msg,
            )
            raise HTTPException(status_code=400, detail=error_msg)

        # 6. ---- AASIST-L DEEPFAKE DETECTION ----
        waveform = preprocessing["waveform"]
        sample_rate = preprocessing["sample_rate"]

        logger.info(
            "[%s] STAGE: AASIST-L detection START "
            "(sample_rate=%d, waveform_shape=%s, duration=%.2fs)",
            request_id,
            sample_rate,
            waveform.shape,
            preprocessing["duration_seconds"],
        )
        t_detect = time.time()

        detection = detect_deepfake_longaudio(waveform, sample_rate)

        detect_ms = (time.time() - t_detect) * 1000
        logger.info(
            "[%s] STAGE: AASIST-L detection END (%.0f ms, success=%s, windows=%s)",
            request_id, detect_ms, detection.get("success"),
            detection.get("num_windows"),
        )

        if not detection.get("success"):
            error_msg = detection.get(
                "error",
                "Deepfake detection failed for an unknown reason.",
            )
            logger.warning(
                "[%s] Detection failed: %s", request_id, error_msg,
            )
            raise HTTPException(status_code=400, detail=error_msg)

        # 7. ---- RISK ENGINE ----
        logger.info("[%s] STAGE: Risk engine START", request_id)
        t_risk = time.time()

        risk_engine = RiskEngine(
            spoof_ratio_threshold=0.15,
            max_spoof_threshold=1.0,
            mean_spoof_threshold=0.1,
        )
        risk_assessment = risk_engine.aggregate(detection["window_scores"])

        risk_ms = (time.time() - t_risk) * 1000
        logger.info(
            "[%s] STAGE: Risk engine END (%.0f ms, prediction=%s, risk=%s)",
            request_id, risk_ms,
            risk_assessment.get("final_prediction"),
            risk_assessment.get("risk_level"),
        )

        if not risk_assessment.get("success"):
            error_msg = risk_assessment.get(
                "error",
                "Risk assessment failed for an unknown reason.",
            )
            logger.warning(
                "[%s] Risk assessment failed: %s", request_id, error_msg,
            )
            raise HTTPException(status_code=400, detail=error_msg)

        # 8. ---- BUILD RESPONSE ----
        file_size_kb = len(file_content) / 1024
        duration_str = f"{preprocessing['duration_seconds']:.2f}s"

        final_prediction = risk_assessment["final_prediction"]
        risk_level = risk_assessment["risk_level"]
        confidence = risk_assessment["confidence"]
        spoof_ratio = risk_assessment["spoof_ratio"]

        result = {
            "verdict": final_prediction,
            "confidence": int(confidence * 100),  # Convert to 0-100 scale
            "risk_level": risk_level,
            "explanation": risk_assessment["assessment"],
            "detection": {
                "model": "AASIST-L (long-audio windowing)",
                "prediction": final_prediction,
                "spoof_ratio": spoof_ratio,
                "num_windows": detection["num_windows"],
                "num_spoof_windows": risk_assessment["num_spoof_windows"],
                "mean_spoof_score": risk_assessment["mean_spoof_score"],
                "mean_bonafide_score": risk_assessment["mean_bonafide_score"],
                "max_spoof_score": risk_assessment["max_spoof_score"],
                "inference_time_ms": detection["inference_time_ms"],
                "device": detection["device"],
            },
            "risk_assessment": {
                "final_prediction": final_prediction,
                "risk_level": risk_level,
                "confidence": confidence,
                "spoof_ratio": spoof_ratio,
                "num_spoof_windows": risk_assessment["num_spoof_windows"],
                "num_total_windows": risk_assessment["num_total_windows"],
                "assessment": risk_assessment["assessment"],
                "strategy_1": risk_assessment["strategy_1"],
                "strategy_2": risk_assessment["strategy_2"],
                "strategy_3": risk_assessment["strategy_3"],
            },
            "preprocessing": {
                "success": True,
                "sample_rate": preprocessing["sample_rate"],
                "channels": preprocessing["channels"],
                "duration_seconds": preprocessing["duration_seconds"],
                "num_samples": preprocessing["num_samples"],
                "format": preprocessing["format"],
            },
        }

        total_ms = (time.time() - t_request) * 1000
        logger.info(
            "[%s] === /api/analyze COMPLETE === verdict=%s  confidence=%d%%  "
            "risk=%s  total=%.0fms (preprocess=%.0f, detect=%.0f, risk=%.0f)",
            request_id, final_prediction, int(confidence * 100),
            risk_level, total_ms, preprocess_ms, detect_ms, risk_ms,
        )

        return result

    except HTTPException:
        # Re-raise our own validation errors
        raise
    except Exception as e:
        total_ms = (time.time() - t_request) * 1000
        logger.exception(
            "[%s] === /api/analyze FAILED === after %.0f ms: %s",
            request_id, total_ms, e,
        )
        # Catch unexpected errors so the frontend gets a clear message
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the audio: {str(e)}",
        )
    finally:
        # Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
