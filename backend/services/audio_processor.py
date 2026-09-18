"""
VoiceShield-AI — Audio Preprocessing Service (Phase 3)

Responsible for:
1. Validating uploaded audio/video files.
2. Running FFmpeg to extract audio, convert to mono PCM WAV.
3. Loading the WAV with librosa to read the waveform & metadata.
4. Returning structured preprocessing results.

This module does NOT perform any AI detection.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
import imageio_ffmpeg

import librosa
import numpy as np

logger = logging.getLogger("voiceshield.audio_processor")

# ---------------------------------------------------------------------------
# Configurable target sample rate.
# Set to None to preserve the original sample rate during FFmpeg conversion.
# A future deepfake-detection model may require a specific rate (e.g. 16000).
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE: int | None = None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def check_ffmpeg() -> bool:
    """Return True if the imageio-ffmpeg executable is available."""
    try:
        return os.path.exists(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return False

def get_ffmpeg_executable() -> str:
    """Return the path to the FFmpeg executable."""
    return imageio_ffmpeg.get_ffmpeg_exe()


# ---------------------------------------------------------------------------
# Dataclass-style result dict builders
# ---------------------------------------------------------------------------

def _success_result(
    *,
    original_filename: str,
    processed_path: str,
    sample_rate: int,
    channels: int,
    duration_seconds: float,
    num_samples: int,
    audio_format: str,
    waveform: np.ndarray,
) -> dict:
    return {
        "success": True,
        "original_filename": original_filename,
        "processed_path": processed_path,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": round(duration_seconds, 4),
        "num_samples": num_samples,
        "format": audio_format,
        "waveform": waveform,
    }


def _error_result(message: str) -> dict:
    return {
        "success": False,
        "error": message,
    }


# ---------------------------------------------------------------------------
# Core preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_audio(
    input_path: str,
    original_filename: str,
    target_sample_rate: int | None = None,
) -> dict:
    """
    Run the full preprocessing pipeline on a single uploaded file.
    """
    import time
    t0 = time.time()

    effective_sr = target_sample_rate if target_sample_rate is not None else TARGET_SAMPLE_RATE

    # --- 1. Basic input validation -------------------------------------------
    logger.info("PREPROCESS 1: input validation START")
    if not input_path or not os.path.isfile(input_path):
        return _error_result("The uploaded file could not be found on the server.")

    if os.path.getsize(input_path) == 0:
        return _error_result("The uploaded file is empty.")
    logger.info("PREPROCESS 1: input validation END")

    # --- 2. Ensure FFmpeg is installed ---------------------------------------
    logger.info("PREPROCESS 2: ffmpeg availability START")
    if not check_ffmpeg():
        return _error_result(
            "FFmpeg is required for audio preprocessing but was not found."
        )
    logger.info("PREPROCESS 2: ffmpeg availability END path=%s", get_ffmpeg_executable())

    # --- 3. Run FFmpeg -------------------------------------------------------
    wav_path = None
    try:
        logger.info("PREPROCESS 3: mkstemp START")
        wav_fd, wav_path = tempfile.mkstemp(
            suffix=".wav",
            prefix=f"voiceshield_proc_{uuid.uuid4().hex[:8]}_",
        )
        os.close(wav_fd)
        logger.info("PREPROCESS 3: mkstemp END path=%s", wav_path)

        logger.info("PREPROCESS 4: FFmpeg START")
        t_ffmpeg = time.time()
        wav_path = _run_ffmpeg(input_path, wav_path, effective_sr)
        ffmpeg_ms = (time.time() - t_ffmpeg) * 1000
        # Re-check and ensure wav_path was indeed generated
        wav_size = os.path.getsize(wav_path)
        logger.info("PREPROCESS 4: FFmpeg END elapsed_ms=%.0fms size=%d bytes", ffmpeg_ms, wav_size)

        # --- 4. Load with librosa --------------------------------------------
        logger.info("PREPROCESS 6: librosa.load START")
        t_librosa = time.time()
        result = _load_with_librosa(wav_path, original_filename)
        librosa_ms = (time.time() - t_librosa) * 1000
        logger.info("PREPROCESS 6: librosa.load END elapsed_ms=%.0fms", librosa_ms)

        total_ms = (time.time() - t0) * 1000
        logger.info("PREPROCESS COMPLETE: total_ms=%.0fms", total_ms)
        return result

    except _PreprocessingError as exc:
        logger.error("Preprocessing failed for %s: %s", original_filename, exc)
        return _error_result(str(exc))
    except Exception as exc:
        logger.exception("Unexpected error while preprocessing %s", original_filename)
        return _error_result(
            f"An unexpected error occurred during audio preprocessing: {str(exc)}"
        )
    finally:
        # Clean up the intermediate WAV created by FFmpeg.
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                logger.warning("Could not delete temp WAV: %s", wav_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _PreprocessingError(Exception):
    """Raised for known/expected preprocessing failures."""


def _run_ffmpeg(input_path: str, output_path: str, sample_rate: int | None) -> str:
    """
    Convert *input_path* to a mono PCM 16-bit WAV at *output_path*.
    """
    cmd: list[str] = [
        get_ffmpeg_executable(),
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-f", "wav",
    ]

    if sample_rate is not None:
        cmd.extend(["-ar", str(sample_rate)])

    cmd.append(output_path)

    # Capture stdout/stderr to log on failure
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=180,  # 3 minutes
        )
    except FileNotFoundError:
        raise _PreprocessingError("FFmpeg executable not found.")
    except subprocess.TimeoutExpired as e:
        logger.error("FFmpeg timed out. Stderr: %s", e.stderr.decode(errors="replace")[:500] if e.stderr else "N/A")
        raise _PreprocessingError("Audio conversion timed out.")
    except Exception as e:
        logger.exception("Unexpected error running FFmpeg command.")
        raise _PreprocessingError(f"Unexpected error running FFmpeg: {str(e)}")

    if proc.returncode != 0:
        stderr_text = proc.stderr.decode(errors="replace").strip()
        logger.error("FFmpeg failed (rc=%d). Stderr: %s", proc.returncode, stderr_text[:1000])
        raise _PreprocessingError(f"FFmpeg failed with exit code {proc.returncode}.")

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise _PreprocessingError("FFmpeg produced an empty file.")

    return output_path


def _load_with_librosa(wav_path: str, original_filename: str) -> dict:
    """
    Load the processed WAV with librosa, validate, and build the result dict.

    Raises ``_PreprocessingError`` on failure.
    """
    try:
        waveform, sr = librosa.load(wav_path, sr=None, mono=True)
    except Exception as exc:
        logger.error("librosa failed to load %s: %s", wav_path, exc)
        raise _PreprocessingError(
            "The processed audio file could not be loaded. "
            "The original file may be corrupted."
        )

    # --- Validation ---
    if waveform is None or waveform.size == 0:
        raise _PreprocessingError(
            "The audio waveform is empty after conversion. "
            "The file may contain no audible audio."
        )

    if not np.all(np.isfinite(waveform)):
        raise _PreprocessingError(
            "The audio waveform contains invalid values (NaN or Inf). "
            "The file may be corrupted."
        )

    duration = float(librosa.get_duration(y=waveform, sr=sr))

    if duration <= 0:
        raise _PreprocessingError(
            "The audio duration is zero or negative. "
            "The file may contain no valid audio."
        )

    return _success_result(
        original_filename=original_filename,
        processed_path=wav_path,
        sample_rate=int(sr),
        channels=1,  # we forced mono via FFmpeg and librosa
        duration_seconds=duration,
        num_samples=int(waveform.shape[0]),
        audio_format="wav",
        waveform=waveform,
    )
