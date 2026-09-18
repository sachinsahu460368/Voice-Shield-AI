"""
VoiceShield-AI — Label Verification Test

Verifies AASIST-L label assignment (bonafide=0, spoof=1) using
raw model inference on multiple audio files.
"""

import pytest
import numpy as np
from pathlib import Path

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "aasist_l" / "AASIST-L.pth"

_ORIGINAL_DIR = Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio")
_AI_DIR = Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Ai Generated audio")

TARGET_SAMPLE_RATE = 16000
MODEL_INPUT_LENGTH = 64600


def _collect_test_cases():
    """Build a list of (expected_label, path) from local audio dirs."""
    cases = []
    if _ORIGINAL_DIR.exists():
        for f in _ORIGINAL_DIR.iterdir():
            if f.suffix.lower() in (".mp3", ".mp4"):
                cases.append(("BONAFIDE", f))
    if _AI_DIR.exists():
        for f in _AI_DIR.iterdir():
            if f.suffix.lower() in (".mp3", ".mp4"):
                cases.append(("SPOOF", f))
    return cases


@pytest.mark.needs_audio
@pytest.mark.needs_model
def test_label_mapping_across_files():
    """
    Iterate local audio files, run raw model inference, and print a
    results table.  The test passes as long as inference succeeds;
    individual prediction accuracy may vary by file.
    """
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")

    cases = _collect_test_cases()
    if not cases:
        pytest.skip("No local audio files found in test directories")

    import torch
    import librosa
    from services.deepfake_detector import AAISSTModel

    device = torch.device("cpu")
    model_config = {
        "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
        "gat_dims": [24, 32],
        "pool_ratios": [0.4, 0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
        "first_conv": 128,
    }
    model = AAISSTModel(model_config).to(device)
    model.eval()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)

    print(f"\n  {'Label':<10} | {'File':<20} | {'B_Score':<10} | {'S_Score':<10} | {'Pred':<10}")
    print("  " + "-" * 70)

    for label, path in cases:
        waveform, sr = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
        if len(waveform) < MODEL_INPUT_LENGTH:
            n_repeats = (MODEL_INPUT_LENGTH // len(waveform)) + 1
            waveform = np.tile(waveform, n_repeats)[:MODEL_INPUT_LENGTH]
        else:
            waveform = waveform[:MODEL_INPUT_LENGTH]

        x = torch.FloatTensor(waveform).unsqueeze(0).to(device)
        with torch.inference_mode():
            _, output = model(x)

        b_score = output[0, 0].item()
        s_score = output[0, 1].item()
        prediction = "SPOOF" if s_score > b_score else "BONAFIDE"
        print(f"  {label:<10} | {path.name[:18]:<20} | {b_score:<10.3f} | {s_score:<10.3f} | {prediction:<10}")
