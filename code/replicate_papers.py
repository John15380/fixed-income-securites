"""Compatibility entry point for verified original-paper numerical checks."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts"/"replicate_literature.py"), run_name="__main__")
