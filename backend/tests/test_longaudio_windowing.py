"""
VoiceShield-AI — Long Audio Windowing Test

Tests the long-audio detection using 4-second overlapping windows.
Ensures the entire audio is analyzed, not just the first 4 seconds.
"""

import pytest
from pathlib import Path

_TEST_CASES = [
    {
        "id": "my_own_voice",
        "name": "My own voice",
        "path": Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\My_own_voice.mp4"),
        "expected": "BONAFIDE",
    },
    {
        "id": "test_audio",
        "name": "Test_Audio",
        "path": Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\Test_Audio.mp3"),
        "expected": "BONAFIDE",
    },
    {
        "id": "ai_video",
        "name": "AI_video",
        "path": Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Ai Generated audio\AI_video.mp3"),
        "expected": "SPOOF",
    },
]


@pytest.mark.needs_audio
@pytest.mark.needs_model
@pytest.mark.parametrize(
    "test_case",
    _TEST_CASES,
    ids=[tc["id"] for tc in _TEST_CASES],
)
def test_longaudio_windowing(test_case):
    """Run detect_deepfake_longaudio and verify the prediction."""
    if not test_case["path"].exists():
        pytest.skip(f"Audio file not found: {test_case['path']}")

    import librosa
    from services.deepfake_detector import detect_deepfake_longaudio

    waveform, sr = librosa.load(str(test_case["path"]), sr=None, mono=True)
    assert waveform is not None and waveform.size > 0

    detection = detect_deepfake_longaudio(waveform, sr)
    assert detection.get("success"), f"Detection failed: {detection.get('error')}"
    assert detection["num_windows"] > 0

    prediction = detection["prediction"]
    print(
        f"\n  {test_case['name']}: prediction={prediction}  "
        f"windows={detection['num_windows']}  "
        f"spoof_ratio={detection['spoof_ratio']*100:.1f}%  "
        f"time={detection['inference_time_ms']:.0f}ms"
    )

    assert prediction == test_case["expected"], (
        f"Expected {test_case['expected']} for {test_case['name']}, got {prediction}"
    )
