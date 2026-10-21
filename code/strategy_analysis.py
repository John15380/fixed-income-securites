"""Compatibility entry point; implementation lives in bond_rp/ and scripts/."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_analysis.py"),
        run_name="__main__",
    )
