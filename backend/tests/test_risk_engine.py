"""
VoiceShield-AI — Risk Engine Comprehensive Test

Tests the windowing → risk engine pipeline:
AASIST-L long-audio windowing → RiskEngine.aggregate → final verdict.
"""

import pytest
from pathlib import Path

_TEST_CASES = [
    {
        "id": "my_own_voice",
        "name": "My own voice",
        "path": Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\My_own_voice.wav"),
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
    {
        "id": "ai_audio2",
        "name": "Ai_audio2",
        "path": Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Ai Generated audio\Ai_audio2.mp3"),
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
def test_risk_engine_pipeline(test_case):
    """Full pipeline: load → detect_deepfake_longaudio → RiskEngine → assert."""
    if not test_case["path"].exists():
        pytest.skip(f"Audio file not found: {test_case['path']}")

    import librosa
    from services.deepfake_detector import detect_deepfake_longaudio
    from services.risk_engine import RiskEngine

    waveform, sr = librosa.load(str(test_case["path"]), sr=None, mono=True)
    assert waveform is not None and waveform.size > 0

    detection = detect_deepfake_longaudio(waveform, sr)
    assert detection.get("success"), f"Detection failed: {detection.get('error')}"

    risk_engine = RiskEngine(
        spoof_ratio_threshold=0.15,
        max_spoof_threshold=1.0,
        mean_spoof_threshold=0.1,
    )
    risk = risk_engine.aggregate(detection["window_scores"])
    assert risk.get("success"), f"Risk engine failed: {risk.get('error')}"

    prediction = risk["final_prediction"]
    print(
        f"\n  {test_case['name']}: verdict={prediction}  "
        f"risk={risk['risk_level']}  confidence={risk['confidence']*100:.0f}%  "
        f"spoof_ratio={risk['spoof_ratio']*100:.1f}%"
    )

    assert prediction == test_case["expected"], (
        f"Expected {test_case['expected']} for {test_case['name']}, "
        f"got {prediction} (spoof_ratio={risk['spoof_ratio']:.3f})"
    )


def test_risk_engine_empty_input():
    """RiskEngine.aggregate with empty input should return success=False."""
    from services.risk_engine import RiskEngine

    engine = RiskEngine()
    result = engine.aggregate([])
    assert result["success"] is False


def test_risk_engine_all_bonafide():
    """All-bonafide windows should produce BONAFIDE verdict with LOW risk."""
    from services.risk_engine import RiskEngine

    window_scores = [
        {"spoof_score": -2.0, "bonafide_score": 3.0, "prediction": "BONAFIDE"},
        {"spoof_score": -1.5, "bonafide_score": 2.5, "prediction": "BONAFIDE"},
        {"spoof_score": -2.5, "bonafide_score": 4.0, "prediction": "BONAFIDE"},
    ]
    engine = RiskEngine(spoof_ratio_threshold=0.15, max_spoof_threshold=1.0, mean_spoof_threshold=0.1)
    result = engine.aggregate(window_scores)

    assert result["success"] is True
    assert result["final_prediction"] == "BONAFIDE"
    assert result["risk_level"] == "LOW"
    assert result["spoof_ratio"] == 0.0


def test_risk_engine_all_spoof():
    """All-spoof windows should produce SPOOF verdict with CRITICAL risk."""
    from services.risk_engine import RiskEngine

    window_scores = [
        {"spoof_score": 3.0, "bonafide_score": -1.0, "prediction": "SPOOF"},
        {"spoof_score": 2.5, "bonafide_score": -0.5, "prediction": "SPOOF"},
        {"spoof_score": 4.0, "bonafide_score": -2.0, "prediction": "SPOOF"},
    ]
    engine = RiskEngine(spoof_ratio_threshold=0.15, max_spoof_threshold=1.0, mean_spoof_threshold=0.1)
    result = engine.aggregate(window_scores)

    assert result["success"] is True
    assert result["final_prediction"] == "SPOOF"
    assert result["risk_level"] == "CRITICAL"
    assert result["spoof_ratio"] == 1.0
