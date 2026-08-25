#!/bin/sh
# mylc0 UCI engine launcher.
#     ./engine/mylc0.sh --weights /abs/path/to/networks/gen_000500.mylc0
# Set MYLC0_PYTHON to pin a specific interpreter (one that has torch).
DIR=$(cd "$(dirname "$0")" && pwd)
exec "${MYLC0_PYTHON:-python}" "$DIR/mylc0.py" "$@"
