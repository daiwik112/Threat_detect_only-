"""
grid/__main__.py — Enables `python -m grid` to call the build CLI.

Usage
-----
python -m grid             # same as python -m grid.build --check
python -m grid --check
python -m grid --theft-km 150 --theft-mw 20 --theft-pf 0.9
python -m grid --load-scale 0.8
"""
from grid.build import _cli

if __name__ == "__main__":
    _cli()
