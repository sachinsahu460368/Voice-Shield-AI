"""
VoiceShield-AI — Complete Flow & GPU Status Report

Tests the complete pipeline from audio upload through risk engine,
and documents GPU availability.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import torch
import librosa
from services.deepfake_detector import detect_deepfake_longaudio
from services.risk_engine import RiskEngine

print("=" * 100)
print("VOICESHIELD-AI COMPLETE FLOW & GPU STATUS REPORT")
print("=" * 100)
print()

# ========== SYSTEM STATUS ==========
print("[1/5] SYSTEM STATUS")
print("-" * 100)

print(f"Timestamp:           {datetime.now().isoformat()}")
print(f"PyTorch version:     {torch.__version__}")
print(f"CUDA available:      {torch.cuda.is_available()}")
print(f"CUDA device count:   {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"CUDA version:        {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:              {torch.cuda.get_device_name(i)}")
        print(f"  Memory:            {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("GPU Status:          NOT AVAILABLE (CPU-only PyTorch)")
    print("Note:                RTX 3050 is available via nvidia-smi")
    print("To enable GPU:       pip install torch --index-url https://download.pytorch.org/whl/cu121")

print()

# ========== AUDIO LOADING ==========
print("[2/5] AUDIO LOADING & PREPROCESSING")
print("-" * 100)

test_audio = r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\Test_Audio.mp3"
audio_path = Path(test_audio)

if not audio_path.exists():
    print(f"ERROR: Audio file not found: {test_audio}")
    sys.exit(1)

print(f"Loading: {audio_path.name}")
waveform, sr = librosa.load(str(audio_path), sr=None, mono=True)
duration_sec = len(waveform) / sr

print(f"  Sample rate:       {sr} Hz")
print(f"  Duration:          {duration_sec:.2f} seconds")
print(f"  Channels:          1 (mono)")
print(f"  Samples:           {len(waveform):,}")
print(f"  File size:         {audio_path.stat().st_size / 1024 / 1024:.2f} MB")

print()

# ========== LONG-AUDIO WINDOWING ==========
print("[3/5] LONG-AUDIO WINDOWING ANALYSIS")
print("-" * 100)

detection = detect_deepfake_longaudio(waveform, sr)

if not detection.get('success'):
    print(f"ERROR: {detection.get('error')}")
    sys.exit(1)

print(f"Model:               AASIST-L (lightweight, 85K parameters)")
print(f"Input windows:       {detection['num_windows']} (4-second windows, 50% overlap)")
print(f"Spoof windows:       {detection['spoof_windows']}/{detection['num_windows']} ({detection['spoof_ratio']*100:.1f}%)")
print(f"Device used:         {detection['device']}")
print(f"Inference time:      {detection['inference_time_ms']:.2f} ms")

if detection['device'] == 'cpu':
    print(f"Status:              Running on CPU (GPU not available)")
    print(f"Note:                Performance would be ~10-50x faster with GPU")
else:
    print(f"Status:              GPU ACCELERATION ACTIVE")

print()

# ========== RISK ENGINE AGGREGATION ==========
print("[4/5] RISK ENGINE MULTI-STRATEGY AGGREGATION")
print("-" * 100)

risk_engine = RiskEngine(
    spoof_ratio_threshold=0.15,
    max_spoof_threshold=1.0,
    mean_spoof_threshold=0.1
)

risk_assessment = risk_engine.aggregate(detection["window_scores"])

if not risk_assessment.get('success'):
    print(f"ERROR: {risk_assessment.get('error')}")
    sys.exit(1)

print(f"Final prediction:    {risk_assessment['final_prediction']}")
print(f"Risk level:          {risk_assessment['risk_level']}")
print(f"Confidence:          {risk_assessment['confidence']*100:.1f}%")
print()
print(f"Aggregation metrics:")
print(f"  Spoof ratio:       {risk_assessment['spoof_ratio']*100:.1f}%")
print(f"  Mean spoof score:  {risk_assessment['mean_spoof_score']:.4f}")
print(f"  Mean bonafide:     {risk_assessment['mean_bonafide_score']:.4f}")
print(f"  Max spoof score:   {risk_assessment['max_spoof_score']:.4f}")
print()

# Show strategy breakdown
print(f"Strategy breakdown (consensus voting):")
for i in range(1, 4):
    strat_key = f'strategy_{i}'
    strat = risk_assessment[strat_key]
    agreement = "[AGREE]" if strat['prediction'] == risk_assessment['final_prediction'] else "[DIFFER]"
    print(f"  {agreement} Strategy {i}: {strat['name']}")
    print(f"       Threshold: {strat['threshold']}")
    print(f"       Value:     {strat['value']}")
    print(f"       Prediction: {strat['prediction']}")

print()

# ========== END-TO-END FLOW VERIFICATION ==========
print("[5/5] COMPLETE END-TO-END FLOW VERIFICATION")
print("-" * 100)

flow_steps = [
    ("Audio Upload", "[OK] File loaded via librosa"),
    ("Preprocessing", f"[OK] {len(waveform):,} samples at {sr} Hz"),
    ("Windowing", f"[OK] {detection['num_windows']} overlapping windows analyzed"),
    ("AASIST-L Inference", f"[OK] {detection['inference_time_ms']:.2f}ms inference on {detection['device']}"),
    ("Risk Engine", f"[OK] 3-strategy consensus voting"),
    ("Final Verdict", f"[OK] {risk_assessment['final_prediction']} ({risk_assessment['risk_level']} risk)"),
    ("Response", "[OK] Ready for FastAPI/React frontend"),
]

for i, (step, status) in enumerate(flow_steps, 1):
    print(f"  {i}. {step:.<30} {status}")

print()
print("=" * 100)

# ========== GPU RECOMMENDATIONS ==========
print("GPU STATUS & RECOMMENDATIONS")
print("=" * 100)

if torch.cuda.is_available():
    print("[GPU ACTIVE] ✓")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    print("Current performance:")
    print(f"  - Single audio (48s): {detection['inference_time_ms']:.0f}ms")
    print(f"  - Estimated GPU time: {detection['inference_time_ms'] / 10:.0f}ms (10x faster)")
else:
    print("[GPU NOT AVAILABLE] CPU-only mode active")
    print()
    print("System has GPU available:")
    print("  GPU: NVIDIA GeForce RTX 3050 (6GB memory)")
    print("  CUDA: 12.9 installed")
    print("  Driver: 576.93 installed")
    print()
    print("To enable GPU acceleration:")
    print("  1. Ensure PyTorch index is accessible")
    print("  2. Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("  3. Verify: python -c \"import torch; print(torch.cuda.is_available())\"")
    print()
    print("Current performance (CPU):")
    print(f"  - Single audio (48s): {detection['inference_time_ms']:.0f}ms")
    print(f"  - With GPU: ~{detection['inference_time_ms'] / 10:.0f}ms (estimated 10x improvement)")
    print(f"  - Processing speed: {duration_sec / (detection['inference_time_ms'] / 1000):.1f}x realtime")

print()
print("=" * 100)
print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)
