"""
VoiceShield-AI — Comprehensive Label Mapping Verification Tests

Converted to proper pytest test functions so that:
  - pytest can collect without crashing
  - missing audio files cause a SKIP, not a sys.exit(1)
  - each test case is independent
"""

import pytest
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Audio fixtures — paths to local test files.
# These are developer-specific and will NOT exist in CI / Render.
# ---------------------------------------------------------------------------

_TEST_CASES = [
    {
        "id": "my_own_voice",
        "name": "My own voice",
        "path": Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\My_own_voice.mp4"),
        "expected": "BONAFIDE",
        "description": "Natural human voice recording (MP4)",
    },
    {
        "id": "test_audio",
        "name": "Test_Audio",
        "path": Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\Test_Audio.mp3"),
        "expected": "BONAFIDE",
        "description": "Natural human voice (48.28 seconds, MP3)",
    },
    {
        "id": "ai_video",
        "name": "AI_video",
        "path": Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Ai Generated audio\AI_video.mp3"),
        "expected": "SPOOF",
        "description": "AI-generated voice (should be SPOOF)",
    },
]


def _audio_available(tc: dict) -> bool:
    return tc["path"].exists()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.needs_audio
@pytest.mark.needs_model
@pytest.mark.parametrize(
    "test_case",
    _TEST_CASES,
    ids=[tc["id"] for tc in _TEST_CASES],
)
def test_label_mapping(test_case):
    """
    Load an audio file, run AASIST-L detect_deepfake, and verify the
    prediction matches the expected label.
    """
    if not _audio_available(test_case):
        pytest.skip(f"Audio file not found: {test_case['path']}")

    import librosa
    from services.deepfake_detector import detect_deepfake

    waveform, sr = librosa.load(str(test_case["path"]), sr=None, mono=True)
    assert waveform is not None and waveform.size > 0, "Loaded waveform is empty"

    detection = detect_deepfake(waveform, sr)
    assert detection.get("success"), f"Detection failed: {detection.get('error')}"

    prediction = detection["prediction"]
    expected = test_case["expected"]

    # Log scores for debugging even on pass
    print(
        f"\n  {test_case['name']}: prediction={prediction}, "
        f"bonafide={detection['bonafide_score']:.6f}, "
        f"spoof={detection['spoof_score']:.6f}, "
        f"time={detection['inference_time_ms']:.2f}ms"
    )

    assert prediction == expected, (
        f"Expected {expected} for {test_case['name']}, got {prediction} "
        f"(bonafide={detection['bonafide_score']:.4f}, "
        f"spoof={detection['spoof_score']:.4f})"
    )
