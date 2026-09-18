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

    Parameters
    ----------
    input_path : str
        Path to the temporary uploaded file on disk.
    original_filename : str
        Original filename as sent by the browser (for logging only).
    target_sample_rate : int | None
        If given, forces FFmpeg to resample to this rate.
        Falls back to the module-level TARGET_SAMPLE_RATE constant.
        If both are None the original sample rate is preserved.

    Returns
    -------
    dict
        A result dict with ``success: True`` and metadata, or
        ``success: False`` and an ``error`` message.
    """
    effective_sr = target_sample_rate if target_sample_rate is not None else TARGET_SAMPLE_RATE

    # --- 1. Basic input validation -------------------------------------------
    if not input_path or not os.path.isfile(input_path):
        return _error_result("The uploaded file could not be found on the server.")

    if os.path.getsize(input_path) == 0:
        return _error_result("The uploaded file is empty.")

    # --- 2. Ensure FFmpeg is installed ---------------------------------------
    if not check_ffmpeg():
        return _error_result(
            "FFmpeg is required for audio preprocessing but was not found. "
            "Please install FFmpeg and make sure it is available on PATH."
        )

    # --- 3. Run FFmpeg -------------------------------------------------------
    wav_path = None
    try:
        wav_fd, wav_path = tempfile.mkstemp(
            suffix=".wav",
            prefix=f"voiceshield_proc_{uuid.uuid4().hex[:8]}_",
        )
        os.close(wav_fd)  # We only need the path; FFmpeg writes the file.

        wav_path = _run_ffmpeg(input_path, wav_path, effective_sr)

        # --- 4. Load with librosa --------------------------------------------
        result = _load_with_librosa(wav_path, original_filename)
        return result

    except _PreprocessingError as exc:
        logger.error("Preprocessing failed for %s: %s", original_filename, exc)
        return _error_result(str(exc))
    except Exception as exc:
        logger.exception("Unexpected error while preprocessing %s", original_filename)
        return _error_result(
            "An unexpected error occurred during audio preprocessing. "
            "Please try again or use a different file."
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

    Raises ``_PreprocessingError`` on failure.
    """
    cmd: list[str] = [
        get_ffmpeg_executable(),
        "-y",               # overwrite output without asking
        "-i", input_path,   # input file
        "-vn",              # discard video stream
        "-ac", "1",         # mono
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-f", "wav",        # output format
    ]

    if sample_rate is not None:
        cmd.extend(["-ar", str(sample_rate)])

    cmd.append(output_path)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,  # generous timeout for big files
        )
    except FileNotFoundError:
        raise _PreprocessingError(
            "FFmpeg is required for audio preprocessing but was not found. "
            "Please install FFmpeg and make sure it is available on PATH."
        )
    except subprocess.TimeoutExpired:
        raise _PreprocessingError(
            "Audio conversion timed out. The file may be too large or corrupted."
        )

    if proc.returncode != 0:
        stderr_text = proc.stderr.decode(errors="replace").strip()
        logger.error("FFmpeg stderr:\n%s", stderr_text)

        # Provide a user-friendly message for common FFmpeg errors.
        lower_err = stderr_text.lower()

        if "does not contain any stream" in lower_err or "no audio" in lower_err:
            raise _PreprocessingError(
                "No audio stream was found in the uploaded video."
            )

        if "invalid data found" in lower_err:
            raise _PreprocessingError(
                "The uploaded file appears to be corrupted or is not a valid audio/video file."
            )

        if "no such file" in lower_err:
            raise _PreprocessingError(
                "The uploaded file could not be located for processing."
            )

        raise _PreprocessingError(
            "FFmpeg was unable to convert the file. "
            "The file may be corrupted or in an unsupported format."
        )

    # Verify that FFmpeg actually produced output.
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise _PreprocessingError(
            "Audio conversion produced an empty file. "
            "The original file may contain no valid audio data."
        )

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
