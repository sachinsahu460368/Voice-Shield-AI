"""
VoiceShield-AI — AASIST-L Standalone Model Test (Phase 5)

Converted to proper pytest: skips gracefully when the checkpoint or
test audio file is missing, no sys.exit() at module level.
"""

import pytest
import numpy as np
from pathlib import Path

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "aasist_l" / "AASIST-L.pth"
TEST_AUDIO_PATH = Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\Test_Audio.mp3")


@pytest.mark.needs_model
@pytest.mark.needs_audio
def test_aasist_l_standalone():
    """Load test audio, run AASIST-L inference, verify output shape."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not TEST_AUDIO_PATH.exists():
        pytest.skip(f"Audio file not found: {TEST_AUDIO_PATH}")

    import torch
    import librosa
    from services.deepfake_detector import AAISSTModel

    TARGET_SAMPLE_RATE = 16000
    MODEL_INPUT_LENGTH = 64600

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

    waveform, sr = librosa.load(str(TEST_AUDIO_PATH), sr=TARGET_SAMPLE_RATE, mono=True)
    assert waveform is not None and waveform.size > 0

    if len(waveform) < MODEL_INPUT_LENGTH:
        n_repeats = (MODEL_INPUT_LENGTH // len(waveform)) + 1
        waveform = np.tile(waveform, n_repeats)[:MODEL_INPUT_LENGTH]
    else:
        waveform = waveform[:MODEL_INPUT_LENGTH]

    x = torch.FloatTensor(waveform).unsqueeze(0).to(device)
    with torch.inference_mode():
        last_hidden, output = model(x)

    # output should be shape (1, 2) — bonafide vs spoof logits
    assert output.shape == (1, 2), f"Unexpected output shape: {output.shape}"

    bonafide_score = output[0, 0].item()
    spoof_score = output[0, 1].item()
    prediction = "SPOOF" if spoof_score > bonafide_score else "BONAFIDE"

    print(
        f"\n  Standalone test: prediction={prediction}, "
        f"bonafide={bonafide_score:.4f}, spoof={spoof_score:.4f}"
    )
