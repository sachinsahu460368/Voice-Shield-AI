"""
VoiceShield-AI Backend — Phase 3 (Audio Preprocessing)

This is the FastAPI server that receives audio files from the React
frontend, validates them, and returns analysis results.

Phase 3 adds a real audio preprocessing pipeline (FFmpeg + librosa).
The AI deepfake detection model has NOT been integrated yet —
the verdict is still a MOCK result.
"""

import logging
import os
import uuid
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.audio_processor import preprocess_audio, check_ffmpeg
from services.deepfake_detector import detect_deepfake_longaudio
from services.risk_engine import RiskEngine

# ---------- Logging ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("voiceshield.main")

# ---------- App setup ----------

app = FastAPI(
    title="VoiceShield-AI",
    description="AI-Powered Voice Deepfake Detection API",
    version="0.1.0",
)

# ---------- CORS ----------
# The Vite dev server runs on http://localhost:5173 by default.
# We allow that origin so the React frontend can call our API.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5173", 
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    import torch
    import sys
    import platform
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "memory_total_mb": None, # Hard to get reliably without psutil
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

    # 1. Check that a file was actually uploaded
    if not audio or not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    # 2. Check MIME type, with extension-based fallback for clients
    #    (e.g. curl) that send application/octet-stream.
    content_type = audio.content_type or ""
    file_ext = os.path.splitext(audio.filename)[1].lower()

    if content_type not in ALLOWED_AUDIO_TYPES:
        # Allow if the Content-Type is generic but the extension is valid
        if content_type == "application/octet-stream" and file_ext in ALLOWED_AUDIO_EXTENSIONS:
            logger.info(
                "Accepted %s via extension fallback (Content-Type: %s, ext: %s)",
                audio.filename, content_type, file_ext,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: Content-Type '{content_type}', "
                       f"extension '{file_ext}'. "
                       f"Please upload an audio file (MP3, WAV, M4A, etc.).",
            )

    # 3. Read file content and check size
    file_content = await audio.read()

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

        # 5. ---- AUDIO PREPROCESSING (Phase 3) ----
        # Run FFmpeg → mono PCM WAV → librosa waveform + metadata.
        preprocessing = preprocess_audio(
            input_path=temp_path,
            original_filename=audio.filename,
        )

        if not preprocessing.get("success"):
            error_msg = preprocessing.get(
                "error",
                "Audio preprocessing failed for an unknown reason.",
            )
            logger.warning("Preprocessing failed for %s: %s", audio.filename, error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # 6. ---- AASIST-L DEEPFAKE DETECTION (Phase 5) ----
        # Run the pretrained AASIST-L anti-spoofing model on the preprocessed waveform.
        waveform = preprocessing["waveform"]
        sample_rate = preprocessing["sample_rate"]

        logger.info(
            "Running AASIST-L long-audio detection on %s "
            "(sample_rate=%d, shape=%s)",
            audio.filename,
            sample_rate,
            waveform.shape,
        )

        # Run long-audio windowing analysis
        logger.info("DEBUG: Calling detect_deepfake_longaudio")
        detection = detect_deepfake_longaudio(waveform, sample_rate)
        logger.info(f"DEBUG: Detection result keys: {detection.keys()}")

        if not detection.get("success"):
            error_msg = detection.get(
                "error",
                "Deepfake detection failed for an unknown reason.",
            )
            logger.warning("Detection failed for %s: %s", audio.filename, error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Run risk engine aggregation on window scores
        risk_engine = RiskEngine(
            spoof_ratio_threshold=0.15,
            max_spoof_threshold=1.0,
            mean_spoof_threshold=0.1
        )
        risk_assessment = risk_engine.aggregate(detection["window_scores"])

        if not risk_assessment.get("success"):
            error_msg = risk_assessment.get(
                "error",
                "Risk assessment failed for an unknown reason.",
            )
            logger.warning("Risk assessment failed for %s: %s", audio.filename, error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # 7. ---- BUILD RESPONSE ----
        # Return the AASIST-L prediction with risk assessment.
        file_size_kb = len(file_content) / 1024
        duration_str = f"{preprocessing['duration_seconds']:.2f}s"

        # Use risk assessment prediction instead of single-window
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

        return result

    except HTTPException:
        # Re-raise our own validation errors
        raise
    except Exception as e:
        # Catch unexpected errors so the frontend gets a clear message
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the audio: {str(e)}",
        )
    finally:
        # 6. Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
