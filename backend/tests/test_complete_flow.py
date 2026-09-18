"""
VoiceShield-AI — Complete Flow & GPU Status Test

Runs the full pipeline: audio load → AASIST-L windowing → risk engine.
Skips when local test audio is absent.
"""

import pytest
from pathlib import Path

TEST_AUDIO = Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\Test_Audio.mp3")


@pytest.mark.needs_audio
@pytest.mark.needs_model
def test_complete_flow():
    """End-to-end pipeline: load → detect_deepfake_longaudio → RiskEngine."""
    if not TEST_AUDIO.exists():
        pytest.skip(f"Audio file not found: {TEST_AUDIO}")

    import librosa
    from services.deepfake_detector import detect_deepfake_longaudio
    from services.risk_engine import RiskEngine

    waveform, sr = librosa.load(str(TEST_AUDIO), sr=None, mono=True)
    assert waveform is not None and waveform.size > 0

    detection = detect_deepfake_longaudio(waveform, sr)
    assert detection.get("success"), f"Detection failed: {detection.get('error')}"
    assert detection["num_windows"] > 0

    risk_engine = RiskEngine(
        spoof_ratio_threshold=0.15,
        max_spoof_threshold=1.0,
        mean_spoof_threshold=0.1,
    )
    risk = risk_engine.aggregate(detection["window_scores"])
    assert risk.get("success"), f"Risk engine failed: {risk.get('error')}"
    assert risk["final_prediction"] in ("BONAFIDE", "SPOOF")
    assert risk["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    print(
        f"\n  Complete flow: verdict={risk['final_prediction']}  "
        f"risk={risk['risk_level']}  confidence={risk['confidence']*100:.0f}%  "
        f"windows={detection['num_windows']}  "
        f"time={detection['inference_time_ms']:.0f}ms"
    )
