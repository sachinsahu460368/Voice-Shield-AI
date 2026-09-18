"""
VoiceShield-AI — pytest configuration

Shared fixtures and marks for the backend test suite.
"""

import sys
from pathlib import Path

import pytest

# Ensure the backend package is importable regardless of how pytest is invoked.
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))


# ---------- Custom marks ----------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_audio: test requires local audio files that may not be present in CI",
    )
    config.addinivalue_line(
        "markers",
        "needs_model: test requires the AASIST-L checkpoint file",
    )
