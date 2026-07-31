"""pycube-lut — load any ``.cube`` / HALD CLUT / 1D tone curve and apply it to a
NumPy image in one call.

Core (NumPy only)::

    from pycube_lut import load_cube, apply_cube

    cube = load_cube("kodak_like.cube")
    graded = apply_cube(image, cube, strength=0.7)

Optional scene-aware selection, for picking from a library of LUTs::

    from pycube_lut import SceneHints, select_luts

    picks = select_luts([("a", cube_a), ("b", cube_b)], SceneHints(warmth="cool"), view, 1)

Reading a HALD CLUT PNG additionally needs Pillow: ``pip install pycube-lut[hald]``.
"""

from .cube import LUT_1D_TO_3D_SIZE, Cube, CubeError, apply_cube, load_cube
from .select import (
    LUT_MAX_CRUSH,
    LUT_PROBE_DIM,
    LutTraits,
    Scene,
    SceneHints,
    characterize_cube,
    score_lut,
    select_luts,
)

__version__ = "0.1.0"

__all__ = [
    "LUT_1D_TO_3D_SIZE",
    "LUT_MAX_CRUSH",
    "LUT_PROBE_DIM",
    "Cube",
    "CubeError",
    "LutTraits",
    "Scene",
    "SceneHints",
    "apply_cube",
    "characterize_cube",
    "load_cube",
    "score_lut",
    "select_luts",
]
