"""
VoiceShield-AI — Comprehensive AASIST-L Label Mapping Verification Test

Tests all three audio files to verify the label mapping fix is correct:
1. My own voice (natural human)
2. Test_Audio.mp3 (natural human)
3. AI_video (AI-generated)
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from services.deepfake_detector import detect_deepfake
import librosa
import numpy as np

print("=" * 80)
print("COMPREHENSIVE AASIST-L LABEL MAPPING VERIFICATION TEST")
print("=" * 80)

# Test files
test_cases = [
    {
        "name": "My own voice",
        "path": r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\My_own_voice.mp4",
        "expected": "BONAFIDE",
        "description": "Natural human voice recording (MP4)"
    },
    {
        "name": "Test_Audio",
        "path": r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\Test_Audio.mp3",
        "expected": "BONAFIDE",
        "description": "Natural human voice (48.28 seconds, MP3)"
    },
    {
        "name": "AI_video",
        "path": r"C:\Users\sachi\Downloads\Smart India Hackethon\Ai Generated audio\AI_video.mp3",
        "expected": "SPOOF",
        "description": "AI-generated voice (should be SPOOF)"
    }
]

results = []

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'=' * 80}")
    print(f"TEST {i}: {test_case['name']}")
    print(f"{'=' * 80}")
    print(f"Description: {test_case['description']}")
    print(f"Path: {test_case['path']}")
    print(f"Expected: {test_case['expected']}")
    print()

    # Check file exists
    audio_path = Path(test_case['path'])
    if not audio_path.exists():
        print(f"[ERROR] File not found: {test_case['path']}")
        results.append({
            'name': test_case['name'],
            'status': 'FAILED',
            'reason': 'File not found'
        })
        continue

    # Load audio
    try:
        print("[1/3] Loading audio...")
        waveform, sr = librosa.load(str(audio_path), sr=None, mono=True)
        print(f"      Sample rate: {sr} Hz")
        print(f"      Duration: {len(waveform) / sr:.2f} seconds")
        print(f"      Samples: {len(waveform)}")
    except Exception as e:
        print(f"[ERROR] Failed to load audio: {e}")
        results.append({
            'name': test_case['name'],
            'status': 'FAILED',
            'reason': f'Load error: {e}'
        })
        continue

    # Run detection
    try:
        print("[2/3] Running AASIST-L detection...")
        detection = detect_deepfake(waveform, sr)

        if not detection.get('success'):
            print(f"[ERROR] Detection failed: {detection.get('error')}")
            results.append({
                'name': test_case['name'],
                'status': 'FAILED',
                'reason': detection.get('error')
            })
            continue

        # Parse results
        print("[3/3] Parsing results...")
        prediction = detection['prediction']
        bonafide_score = detection['bonafide_score']
        spoof_score = detection['spoof_score']
        inference_time = detection['inference_time_ms']
        device = detection['device']

        print()
        print("RESULTS:")
        print(f"  Prediction:        {prediction}")
        print(f"  Bonafide score:    {bonafide_score:.6f}")
        print(f"  Spoof score:       {spoof_score:.6f}")
        print(f"  Score difference:  {abs(bonafide_score - spoof_score):.6f}")
        print(f"  Inference time:    {inference_time:.2f} ms")
        print(f"  Device:            {device}")

        # Check if prediction matches expected
        if prediction == test_case['expected']:
            print(f"\n[PASS] Prediction matches expected: {test_case['expected']}")
            status = 'PASS'
        else:
            print(f"\n[FAIL] Expected {test_case['expected']}, got {prediction}")
            status = 'FAIL'

        results.append({
            'name': test_case['name'],
            'status': status,
            'prediction': prediction,
            'expected': test_case['expected'],
            'bonafide_score': bonafide_score,
            'spoof_score': spoof_score,
            'inference_time_ms': inference_time,
            'device': device
        })

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'name': test_case['name'],
            'status': 'FAILED',
            'reason': str(e)
        })

# Final summary
print(f"\n{'=' * 80}")
print("FINAL SUMMARY")
print(f"{'=' * 80}\n")

passed = sum(1 for r in results if r['status'] == 'PASS')
failed = sum(1 for r in results if r['status'] in ['FAIL', 'FAILED'])

print(f"Total tests: {len(results)}")
print(f"Passed:      {passed}")
print(f"Failed:      {failed}")
print()

for r in results:
    if r['status'] in ['PASS', 'FAIL']:
        match_mark = "[OK]" if r['status'] == 'PASS' else "[XX]"
        print(f"{match_mark} {r['name']}")
        print(f"     Expected:  {r['expected']}")
        print(f"     Got:       {r['prediction']}")
        print(f"     Bonafide:  {r['bonafide_score']:.6f}")
        print(f"     Spoof:     {r['spoof_score']:.6f}")
    else:
        print(f"[ERROR] {r['name']}: {r.get('reason', 'Unknown error')}")
    print()

if failed == 0 and passed == 3:
    print("=" * 80)
    print("ALL TESTS PASSED - LABEL MAPPING IS CORRECT")
    print("AI voice detection is working correctly!")
    print("Ready for FastAPI and React frontend validation")
    print("=" * 80)
    sys.exit(0)
else:
    print("=" * 80)
    print("SOME TESTS FAILED - CHECK RESULTS ABOVE")
    print("=" * 80)
    sys.exit(1)
