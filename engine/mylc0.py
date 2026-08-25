"""mylc0 UCI engine entry point.

    python engine/mylc0.py --weights networks/gen_000500.mylc0

On Windows ``engine\\mylc0.bat`` wraps this so a GUI can point straight at it.
The engine only needs PyTorch and python-chess: no training code, no data, no
configuration file -- just the network file.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from mylc0.engine.uci import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
