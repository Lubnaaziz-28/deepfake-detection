"""Entry point so the package can also run with `python -m deepfake_scan`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
