"""Classify exported flows using the supported artifact and feature contract.

The 2024 packet capture prototype was retired: it substituted packet fields for
bidirectional flow statistics and did not restore the training scaler. See
../docs/AUDIT.md and ../docs/ARCHITECTURE.md before implementing a live sensor.
"""

import sys

from netattdetector.cli import main

if __name__ == "__main__":
    main(["predict", *sys.argv[1:]])
