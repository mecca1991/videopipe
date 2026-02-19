# tests/test_cli.py
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_infer_script_runs_with_help():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "infer.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "image" in result.stdout.lower()


def test_train_script_runs_with_help():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "train.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "train" in result.stdout.lower()
