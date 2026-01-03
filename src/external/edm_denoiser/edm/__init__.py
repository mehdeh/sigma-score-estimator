"""
EDM (Elucidating the Design Space of Diffusion Models) external dependencies.

This package contains code from the NVlabs/edm repository:
https://github.com/NVlabs/edm

License: CC BY-NC-SA 4.0
See LICENSE.txt and NOTICE.txt in this directory for details.
"""

# Make dnnlib and torch_utils importable from this package
import sys
import os

# Add this directory to the path so imports work
_edm_dir = os.path.dirname(os.path.abspath(__file__))
if _edm_dir not in sys.path:
    sys.path.insert(0, _edm_dir)

