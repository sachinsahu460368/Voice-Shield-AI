import sys
from pathlib import Path
import numpy as np
import torch
import librosa

# Add backend to path so we can import the model
backend_dir = Path(r"C:\Users\sachi\OneDrive\Documents\VoiceShield-AI-main\backend")
sys.path.insert(0, str(backend_dir))

from services.deepfake_detector import AAISSTModel, SincConv, ResidualBlock, GraphAttentionLayer, HtrgGraphAttentionLayer, GraphPool

# Configuration
CHECKPOINT_PATH = Path(r"C:\Users\sachi\OneDrive\Documents\VoiceShield-AI-main\backend\models\aasist_l\AASIST-L.pth")
TARGET_SAMPLE_RATE = 16000
MODEL_INPUT_LENGTH = 64600

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model_args = {
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
    "first_conv": 128,
}
model = AAISSTModel(model_args).to(device)
model.eval()
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint)

# Test cases
test_cases = []
dir_orig = Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio")
for f in dir_orig.iterdir():
    if f.suffix in [".mp3", ".mp4"]:
        test_cases.append(("BONAFIDE", f))

dir_ai = Path(r"C:\Users\sachi\Downloads\Smart India Hackethon\Ai Generated audio")
for f in dir_ai.iterdir():
    if f.suffix in [".mp3", ".mp4"]:
        test_cases.append(("SPOOF", f))

# Inference
def run_inference(audio_path):
    waveform, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
    if len(waveform) < MODEL_INPUT_LENGTH:
        n_repeats = (MODEL_INPUT_LENGTH // len(waveform)) + 1
        waveform = np.tile(waveform, n_repeats)[:MODEL_INPUT_LENGTH]
    else:
        waveform = waveform[:MODEL_INPUT_LENGTH]
    x = torch.FloatTensor(waveform).unsqueeze(0).to(device)
    with torch.inference_mode():
        _, output = model(x)
    return output

print(f"| {'Label':<10} | {'File':<20} | {'B_Score':<10} | {'S_Score':<10} | {'Pred':<10} |")
print("-" * 75)
for label, path in test_cases:
    output = run_inference(path)
    b_score = output[0, 0].item()
    s_score = output[0, 1].item()
    prediction = "SPOOF" if s_score > b_score else "BONAFIDE"
    print(f"| {label:<10} | {path.name[:18]:<20} | {b_score:<10.3f} | {s_score:<10.3f} | {prediction:<10} |")
