"""
VoiceShield-AI — Long Audio Windowing Test

Tests the new long-audio detection using overlapping windows.
This ensures the entire audio is analyzed, not just the first 4 seconds.
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from services.deepfake_detector import detect_deepfake_longaudio
import librosa

print("=" * 80)
print("LONG AUDIO WINDOWING TEST - AASIST-L")
print("=" * 80)

# Test files
test_cases = [
    {
        "name": "My own voice",
        "path": r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\My_own_voice.mp4",
        "expected": "BONAFIDE",
        "description": "Natural human voice (314 sec) - FULL AUDIO ANALYSIS"
    },
    {
        "name": "Test_Audio",
        "path": r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\Test_Audio.mp3",
        "expected": "BONAFIDE",
        "description": "Natural human voice (48 sec)"
    },
    {
        "name": "AI_video",
        "path": r"C:\Users\sachi\Downloads\Smart India Hackethon\Ai Generated audio\AI_video.mp3",
        "expected": "SPOOF",
        "description": "AI-generated voice (20 sec) - FULL AUDIO ANALYSIS"
    }
]

results = []

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'=' * 80}")
    print(f"TEST {i}: {test_case['name']}")
    print(f"{'=' * 80}")
    print(f"Description: {test_case['description']}")
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
        print(f"      Samples: {len(waveform):,}")
    except Exception as e:
        print(f"[ERROR] Failed to load audio: {e}")
        results.append({
            'name': test_case['name'],
            'status': 'FAILED',
            'reason': f'Load error: {e}'
        })
        continue

    # Run long-audio detection
    try:
        print("[2/3] Running AASIST-L long-audio windowing analysis...")
        detection = detect_deepfake_longaudio(waveform, sr)

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
        num_windows = detection['num_windows']
        spoof_windows = detection['spoof_windows']
        spoof_ratio = detection['spoof_ratio']
        inference_time = detection['inference_time_ms']
        device = detection['device']
        total_duration = detection['total_duration_sec']

        print()
        print("LONG-AUDIO ANALYSIS RESULTS:")
        print(f"  Total duration:        {total_duration:.2f} seconds")
        print(f"  Windows analyzed:      {num_windows} (50% overlap)")
        print(f"  Spoof windows:         {spoof_windows}/{num_windows} ({spoof_ratio*100:.1f}%)")
        print(f"  Bonafide windows:      {num_windows - spoof_windows}/{num_windows}")
        print()
        print(f"  Aggregate prediction:  {prediction}")
        print(f"  Mean bonafide score:   {bonafide_score:.6f}")
        print(f"  Mean spoof score:      {spoof_score:.6f}")
        print(f"  Score difference:      {abs(bonafide_score - spoof_score):.6f}")
        print(f"  Inference time:        {inference_time:.2f} ms")
        print(f"  Device:                {device}")

        # Show window breakdown
        if num_windows <= 10:
            print()
            print("  WINDOW BREAKDOWN:")
            for wr in detection['window_scores']:
                pred_mark = "[SPOOF]" if wr['prediction'] == "SPOOF" else "[BONAFIDE]"
                print(f"    Window {wr['window']:2d}: {pred_mark} "
                      f"bonafide={wr['bonafide_score']:7.3f}, "
                      f"spoof={wr['spoof_score']:7.3f}")

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
            'num_windows': num_windows,
            'spoof_windows': spoof_windows,
            'spoof_ratio': spoof_ratio,
            'inference_time_ms': inference_time,
            'device': device,
            'total_duration_sec': total_duration
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
print("FINAL SUMMARY - LONG AUDIO WINDOWING")
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
        print(f"     Duration:     {r['total_duration_sec']:.2f} sec")
        print(f"     Windows:       {r['num_windows']} (spoof: {r['spoof_windows']}, ratio: {r['spoof_ratio']*100:.1f}%)")
        print(f"     Expected:      {r['expected']}")
        print(f"     Got:           {r['prediction']}")
        print(f"     Mean Bonafide: {r['bonafide_score']:.6f}")
        print(f"     Mean Spoof:    {r['spoof_score']:.6f}")
        print(f"     Time:          {r['inference_time_ms']:.2f} ms")
    else:
        print(f"[ERROR] {r['name']}: {r.get('reason', 'Unknown error')}")
    print()

if failed == 0 and passed == 3:
    print("=" * 80)
    print("ALL TESTS PASSED - LONG AUDIO WINDOWING WORKING CORRECTLY")
    print("Full audio is now being analyzed (not just first 4 seconds)")
    print("=" * 80)
    sys.exit(0)
else:
    print("=" * 80)
    print(f"TEST RESULTS: {passed} PASS, {failed} FAIL")
    print("=" * 80)
    sys.exit(1)
