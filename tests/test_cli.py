# tests/test_cli.py
import subprocess
import sys
import pytest


def test_infer_script_runs_with_help():
    result = subprocess.run(
        [sys.executable, "infer.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "image" in result.stdout.lower()


def test_train_script_runs_with_help():
    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "train" in result.stdout.lower()
