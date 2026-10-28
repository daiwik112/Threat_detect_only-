# conftest.py — root pytest configuration for sharavathi-theft project.
# Ensures the project root (sharavathi-theft/) is on sys.path so that
# `from grid.xxx import ...` works without installing the package.

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
