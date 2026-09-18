
import os
import sys
from pathlib import Path

# Add backend to path so we can import services
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from services.audio_processor import preprocess_audio

def test_preprocess():
    # Use a dummy file that exists
    input_file = backend_dir / "debug.wav"
    if not input_file.exists():
        print(f"Debugger file not found at {input_file}")
        return

    print(f"Testing preprocessing with {input_file}")

    result = preprocess_audio(str(input_file), "debug.wav")

    if result["success"]:
        print("Preprocessing SUCCESS")
        print(f"Duration: {result['duration_seconds']}s")
        print(f"Format: {result['format']}")
    else:
        print(f"Preprocessing FAILED: {result['error']}")

if __name__ == "__main__":
    test_preprocess()
