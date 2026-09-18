import os
import json
import csv
import numpy as np
import librosa
from pathlib import Path
import logging

# Assuming the project structure and service locations per inspection
from backend.services.deepfake_detector import detect_deepfake_longaudio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("language_evaluation")

def run_evaluation(data_dir: str, output_dir: str):
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    # Iterate over languages
    for lang_dir in data_path.iterdir():
        if not lang_dir.is_dir():
            continue

        lang = lang_dir.name
        logger.info(f"Evaluating language: {lang}")

        for category in ['real', 'spoof']:
            cat_dir = lang_dir / category
            if not cat_dir.exists():
                continue

            for audio_file in cat_dir.iterdir():
                if audio_file.suffix not in ['.wav', '.mp3', '.m4a']:
                    continue

                logger.info(f"Running detection on {audio_file.name} ({lang}/{category})")

                # Preprocessing
                waveform, sr = librosa.load(str(audio_file), sr=16000)

                # Detection
                detection = detect_deepfake_longaudio(waveform, sr)

                if detection.get("success"):
                    results.append({
                        "language": lang,
                        "file": audio_file.name,
                        "category": category,
                        "verdict": detection["prediction"],
                        "spoof_score": detection["spoof_score"],
                        "bonafide_score": detection["bonafide_score"],
                        "spoof_ratio": detection["spoof_ratio"],
                        "error": None
                    })
                else:
                    logger.error(f"Failed to detect {audio_file.name}: {detection.get('error')}")
                    results.append({
                        "language": lang,
                        "file": audio_file.name,
                        "category": category,
                        "verdict": "ERROR",
                        "spoof_score": None,
                        "bonafide_score": None,
                        "spoof_ratio": None,
                        "error": detection.get("error")
                    })

    # Save results
    with open(output_path / "evaluation_results.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["language", "file", "category", "verdict", "spoof_score", "bonafide_score", "spoof_ratio", "error"])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Evaluation complete. Results saved to {output_path / 'evaluation_results.csv'}")

if __name__ == "__main__":
    # Placeholder for running the script.
    # The evaluation requires a structured data directory to be present.
    # Users should populate language_test_data/ first.

    data_dir = "language_test_data"
    output_dir = "evaluation_outputs"

    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Please populate it with language-organized audio samples.")
    else:
        run_evaluation(data_dir, output_dir)
