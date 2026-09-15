"""
VoiceShield-AI — Risk Engine Comprehensive Test

Tests the complete pipeline with risk engine aggregation:
AASIST-L windowing → Risk Engine → Final Verdict
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from services.deepfake_detector import detect_deepfake_longaudio
from services.risk_engine import RiskEngine
import librosa

print("=" * 100)
print("RISK ENGINE COMPREHENSIVE TEST - MULTI-STRATEGY AGGREGATION")
print("=" * 100)

# Initialize risk engine with thresholds
risk_engine = RiskEngine(
    spoof_ratio_threshold=0.15,  # 15% of windows must be spoof
    max_spoof_threshold=1.0,     # lowered from 2.0
    mean_spoof_threshold=0.1     # lowered from 0.5
)

# Test files
test_cases = [
    {
        "name": "My own voice",
        "path": r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\My_own_voice.wav",
        "expected": "BONAFIDE",
        "description": "Natural human voice (314 sec)"
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
        "description": "AI-generated voice (20 sec)"
    },
    {
        "name": "Ai_audio2",
        "path": r"C:\Users\sachi\Downloads\Smart India Hackethon\Ai Generated audio\Ai_audio2.mp3",
        "expected": "SPOOF",
        "description": "AI-generated voice (alternate sample)"
    }
]

results = []

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'=' * 100}")
    print(f"TEST {i}: {test_case['name']}")
    print(f"{'=' * 100}")
    print(f"Description: {test_case['description']}")
    print(f"Expected:    {test_case['expected']}")
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
        print("[STEP 1/4] Loading audio...")
        waveform, sr = librosa.load(str(audio_path), sr=None, mono=True)
        print(f"           Sample rate: {sr} Hz")
        print(f"           Duration: {len(waveform) / sr:.2f} seconds")
        print(f"           Samples: {len(waveform):,}")
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
        print("[STEP 2/4] Running AASIST-L long-audio windowing...")
        detection = detect_deepfake_longaudio(waveform, sr)

        if not detection.get('success'):
            print(f"[ERROR] Detection failed: {detection.get('error')}")
            results.append({
                'name': test_case['name'],
                'status': 'FAILED',
                'reason': detection.get('error')
            })
            continue

        print(f"           {detection['num_windows']} windows analyzed")

        # Run risk engine aggregation
        print("[STEP 3/4] Running Risk Engine aggregation...")
        risk_assessment = risk_engine.aggregate(detection['window_scores'])

        if not risk_assessment.get('success'):
            print(f"[ERROR] Risk assessment failed: {risk_assessment.get('error')}")
            results.append({
                'name': test_case['name'],
                'status': 'FAILED',
                'reason': risk_assessment.get('error')
            })
            continue

        # Parse results
        print("[STEP 4/4] Generating final verdict...")
        print()

        prediction = risk_assessment['final_prediction']
        risk_level = risk_assessment['risk_level']
        confidence = risk_assessment['confidence']
        spoof_ratio = risk_assessment['spoof_ratio']
        mean_spoof = risk_assessment['mean_spoof_score']
        mean_bonafide = risk_assessment['mean_bonafide_score']
        assessment = risk_assessment['assessment']

        print("=" * 72)
        print("RISK ENGINE RESULTS")
        print("=" * 72)
        print()
        print(f"  FINAL VERDICT:         {prediction}")
        print(f"  RISK LEVEL:            {risk_level}")
        print(f"  CONFIDENCE:            {confidence*100:.1f}%")
        print()
        print("  AGGREGATION METRICS:")
        print(f"    Spoof Ratio:         {spoof_ratio*100:.1f}% ({risk_assessment['num_spoof_windows']}/{risk_assessment['num_total_windows']} windows)")
        print(f"    Mean Spoof Score:    {mean_spoof:.4f}")
        print(f"    Mean Bonafide Score: {mean_bonafide:.4f}")
        print(f"    Max Spoof Score:     {risk_assessment['max_spoof_score']:.4f}")
        print()
        print("  STRATEGY BREAKDOWN:")
        for strategy_key in ['strategy_1', 'strategy_2', 'strategy_3']:
            strat = risk_assessment[strategy_key]
            pred_mark = "[AGREE]" if strat['prediction'] == prediction else "[DIFF]"
            print(f"    {pred_mark} {strat['name']}")
            print(f"       Threshold: {strat['threshold']}")
            print(f"       Value:     {strat['value']}")
            print(f"       Prediction: {strat['prediction']} (confidence: {strat['confidence']*100:.1f}%)")
        print()
        print(f"  ASSESSMENT:")
        print(f"    {assessment}")
        print()

        # Check if prediction matches expected
        if prediction == test_case['expected']:
            print(f"  [PASS] Prediction matches expected: {test_case['expected']}")
            status = 'PASS'
        else:
            print(f"  [FAIL] Expected {test_case['expected']}, got {prediction}")
            status = 'FAIL'

        results.append({
            'name': test_case['name'],
            'status': status,
            'prediction': prediction,
            'expected': test_case['expected'],
            'risk_level': risk_level,
            'confidence': confidence,
            'spoof_ratio': spoof_ratio,
            'assessment': assessment
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
print(f"\n{'=' * 100}")
print("FINAL SUMMARY - RISK ENGINE TEST")
print(f"{'=' * 100}\n")

passed = sum(1 for r in results if r['status'] == 'PASS')
failed = sum(1 for r in results if r['status'] in ['FAIL', 'FAILED'])

print(f"Total tests:    {len(results)}")
print(f"Passed:         {passed}")
print(f"Failed:         {failed}")
print()

# Summary table
print("+" + "-" * 20 + "+" + "-" * 14 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 16 + "+")
print("| Audio File       | Expected     | Predicted | Risk      | Status         |")
print("+" + "-" * 20 + "+" + "-" * 14 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 16 + "+")

for r in results:
    if r['status'] in ['PASS', 'FAIL']:
        status_mark = "PASS" if r['status'] == 'PASS' else "FAIL"
        name = r['name'][:18]
        expected = r['expected']
        predicted = r['prediction']
        risk = r['risk_level']
        print(f"| {name:18} | {expected:12} | {predicted:9} | {risk:9} | {status_mark:14} |")
    else:
        name = r['name'][:18]
        reason = r.get('reason', 'Unknown')[:27]
        print(f"| {name:18} | ERROR:       | {reason:27} |")

print("+" + "-" * 20 + "+" + "-" * 14 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 16 + "+")
print()

if failed == 0 and passed == 4:
    print("=" * 100)
    print("[SUCCESS] ALL TESTS PASSED - RISK ENGINE WORKING CORRECTLY")
    print("Multi-strategy aggregation successfully identifying genuine vs AI-generated audio")
    print("=" * 100)
    sys.exit(0)
else:
    print("=" * 100)
    print(f"RESULTS: {passed} PASS, {failed} FAIL")
    print("=" * 100)
    sys.exit(1)
