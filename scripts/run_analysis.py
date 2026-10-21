"""Run the corrected full analysis through the single canonical report entry."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_readme_results import main

if __name__ == "__main__":
    main(default_full=True)
