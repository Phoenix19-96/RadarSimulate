"""Run the editable FMCW example from a source checkout or editable install."""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from configs.fmcw_example import CONFIG, OUTPUT
from radarsim.simulation import run_and_save


if __name__ == "__main__":
    result, directory = run_and_save(CONFIG, OUTPUT)
    print(f"Saved {len(result.detections)} detections to {directory}")
