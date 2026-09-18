"""
VoiceShield-AI — FastAPI Integration Test with Risk Engine

Tests that the /api/analyze endpoint correctly uses long-audio
windowing, the risk engine, and returns the proper response structure.
"""

import pytest
from pathlib import Path


_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
    ".m4a": "audio/mp4",
}

_TEST_CASES = [
    {
        "id": "natural_test_audio",
        "name": "Test_Audio (Natural)",
        "path": Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\Test_Audio.mp3"),
        "expected": "BONAFIDE",
    },
    {
        "id": "ai_video",
        "name": "AI_video (AI-Generated)",
        "path": Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Ai Generated audio\AI_video.mp3"),
        "expected": "SPOOF",
    },
]


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


@pytest.mark.needs_audio
@pytest.mark.needs_model
@pytest.mark.parametrize(
    "test_case",
    _TEST_CASES,
    ids=[tc["id"] for tc in _TEST_CASES],
)
def test_analyze_endpoint(client, test_case):
    """POST to /api/analyze and verify response structure and verdict."""
    audio_path = test_case["path"]
    if not audio_path.exists():
        pytest.skip(f"Audio file not found: {audio_path}")

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    ext = audio_path.suffix.lower()
    mime = _MIME_TYPES.get(ext, "audio/mpeg")

    response = client.post(
        "/api/analyze",
        files={"audio": (audio_path.name, audio_data, mime)},
    )
    assert response.status_code == 200, f"HTTP {response.status_code}: {response.text}"

    data = response.json()

    # Verify required top-level keys
    for key in ("verdict", "risk_level", "confidence", "explanation",
                "detection", "risk_assessment", "preprocessing"):
        assert key in data, f"Missing key in response: {key}"

    # Check verdict matches expected
    verdict = data["verdict"]
    print(
        f"\n  {test_case['name']}: verdict={verdict}  "
        f"risk={data['risk_level']}  confidence={data['confidence']}%"
    )

    assert verdict == test_case["expected"], (
        f"Expected {test_case['expected']}, got {verdict}"
    )


def test_health_endpoint(client):
    """GET /api/health should return 200."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_analyze_no_file(client):
    """POST /api/analyze with no file should return 422 (validation error)."""
    response = client.post("/api/analyze")
    assert response.status_code == 422


def test_analyze_empty_file(client):
    """POST /api/analyze with an empty file should return 400."""
    response = client.post(
        "/api/analyze",
        files={"audio": ("empty.mp3", b"", "audio/mpeg")},
    )
    assert response.status_code == 400
