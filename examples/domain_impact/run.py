"""Run the isolated C2 domain-impact solver snapshot."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
VENDOR=HERE/"vendor"
# Keep this vendor snapshot ahead of the public mpm_jax/examples packages.
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(VENDOR))

from run_metal_tl_fresh import main

if __name__ == "__main__":
    main()
