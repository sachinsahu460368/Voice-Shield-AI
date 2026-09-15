"""
VoiceShield-AI — FastAPI Integration Test with Risk Engine

Tests that the /api/analyze endpoint correctly uses:
1. Long-audio windowing (detect_deepfake_longaudio)
2. Risk engine aggregation
3. Returns proper response structure
"""

import sys
from pathlib import Path
import json

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from main import app

print("=" * 100)
print("FASTAPI INTEGRATION TEST - RISK ENGINE ENDPOINT")
print("=" * 100)

client = TestClient(app)

# Test files
test_cases = [
    {
        "name": "Test_Audio (Natural)",
        "path": r"C:\Users\sachi\Downloads\Smart India Hackethon\Original Audio\Test_Audio.mp3",
        "expected": "BONAFIDE",
    },
    {
        "name": "AI_video (AI-Generated)",
        "path": r"C:\Users\sachi\Downloads\Smart India Hackethon\Ai Generated audio\AI_video.mp3",
        "expected": "SPOOF",
    },
]

results = []

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'=' * 100}")
    print(f"TEST {i}: {test_case['name']}")
    print(f"{'=' * 100}")

    audio_path = Path(test_case['path'])
    if not audio_path.exists():
        print(f"[ERROR] File not found: {test_case['path']}")
        results.append({
            'name': test_case['name'],
            'status': 'FAILED',
            'reason': 'File not found'
        })
        continue

    try:
        # Read audio file
        with open(audio_path, 'rb') as f:
            audio_data = f.read()

        # Determine MIME type from extension
        ext = audio_path.suffix.lower()
        mime_types = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.mp4': 'video/mp4',
            '.m4a': 'audio/mp4',
        }
        mime_type = mime_types.get(ext, 'audio/mpeg')

        print(f"[1/2] Uploading {audio_path.name} ({len(audio_data) / 1024:.1f} KB)...")

        # Make request to /api/analyze
        response = client.post(
            "/api/analyze",
            files={"audio": (audio_path.name, audio_data, mime_type)},
        )

        print(f"[2/2] Processing response...")
        print()

        if response.status_code != 200:
            print(f"[ERROR] HTTP {response.status_code}: {response.text}")
            results.append({
                'name': test_case['name'],
                'status': 'FAILED',
                'reason': f"HTTP {response.status_code}"
            })
            continue

        # Parse response
        data = response.json()

        # Display results
        verdict = data.get('verdict')
        risk_level = data.get('risk_level')
        confidence = data.get('confidence')
        explanation = data.get('explanation')

        print("ENDPOINT RESPONSE:")
        print(f"  Verdict:      {verdict}")
        print(f"  Risk Level:   {risk_level}")
        print(f"  Confidence:   {confidence}%")
        print(f"  Explanation:  {explanation}")
        print()

        # Check detection details
        detection = data.get('detection', {})
        print("DETECTION DETAILS:")
        print(f"  Model:                 {detection.get('model')}")
        print(f"  Windows Analyzed:      {detection.get('num_windows')}")
        print(f"  Spoof Windows:         {detection.get('num_spoof_windows')}")
        print(f"  Spoof Ratio:           {detection.get('spoof_ratio')*100:.1f}%")
        print(f"  Mean Spoof Score:      {detection.get('mean_spoof_score'):.4f}")
        print(f"  Mean Bonafide Score:   {detection.get('mean_bonafide_score'):.4f}")
        print(f"  Max Spoof Score:       {detection.get('max_spoof_score'):.4f}")
        print(f"  Inference Time:        {detection.get('inference_time_ms'):.2f}ms")
        print(f"  Device:                {detection.get('device')}")
        print()

        # Check risk assessment strategies
        risk_assessment = data.get('risk_assessment', {})
        if 'strategy_1' in risk_assessment:
            print("RISK ENGINE STRATEGIES:")
            for i in range(1, 4):
                strat_key = f'strategy_{i}'
                if strat_key in risk_assessment:
                    strat = risk_assessment[strat_key]
                    print(f"  Strategy {i}: {strat.get('name')}")
                    print(f"    Threshold: {strat.get('threshold')}")
                    print(f"    Value:     {strat.get('value')}")
                    print(f"    Prediction: {strat.get('prediction')}")
            print()

        # Check if verdict matches expected
        if verdict == test_case['expected']:
            print(f"[PASS] Verdict matches expected: {test_case['expected']}")
            status = 'PASS'
        else:
            print(f"[FAIL] Expected {test_case['expected']}, got {verdict}")
            status = 'FAIL'

        # Verify response structure
        required_keys = ['verdict', 'risk_level', 'confidence', 'explanation', 'detection', 'risk_assessment']
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            print(f"[WARN] Missing response keys: {missing_keys}")

        results.append({
            'name': test_case['name'],
            'status': status,
            'verdict': verdict,
            'expected': test_case['expected'],
            'risk_level': risk_level,
            'confidence': confidence,
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
print("FINAL SUMMARY - FASTAPI INTEGRATION TEST")
print(f"{'=' * 100}\n")

passed = sum(1 for r in results if r['status'] == 'PASS')
failed = sum(1 for r in results if r['status'] in ['FAIL', 'FAILED'])

print(f"Total tests:    {len(results)}")
print(f"Passed:         {passed}")
print(f"Failed:         {failed}")
print()

# Summary table
print("+" + "-" * 30 + "+" + "-" * 14 + "+" + "-" * 11 + "+" + "-" * 16 + "+")
print("| Audio File                 | Expected     | Predicted | Status         |")
print("+" + "-" * 30 + "+" + "-" * 14 + "+" + "-" * 11 + "+" + "-" * 16 + "+")

for r in results:
    if r['status'] in ['PASS', 'FAIL']:
        status_mark = "PASS" if r['status'] == 'PASS' else "FAIL"
        name = r['name'][:28]
        expected = r['expected']
        predicted = r['verdict']
        print(f"| {name:28} | {expected:12} | {predicted:9} | {status_mark:14} |")
    else:
        name = r['name'][:28]
        reason = r.get('reason', 'Unknown')[:27]
        print(f"| {name:28} | ERROR:       | {reason:27} |")

print("+" + "-" * 30 + "+" + "-" * 14 + "+" + "-" * 11 + "+" + "-" * 16 + "+")
print()

if failed == 0 and passed == len(results):
    print("=" * 100)
    print("[SUCCESS] ALL TESTS PASSED - FASTAPI INTEGRATION WORKING CORRECTLY")
    print("Risk engine properly integrated with /api/analyze endpoint")
    print("=" * 100)
    sys.exit(0)
else:
    print("=" * 100)
    print(f"RESULTS: {passed} PASS, {failed} FAIL")
    print("=" * 100)
    sys.exit(1)
