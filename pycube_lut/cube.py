"""Read an Adobe ``.cube`` 3D LUT and apply it to a NumPy image.

A 3D lookup table is the honest way to carry a film-style colour transform.
Global knobs -- saturation, an S-curve, white balance, gamma, a black point --
move every hue the same way; a real stock's signature is hue-selective and
needs a full three-dimensional lookup to reproduce.

Video tools apply these tables through ffmpeg's ``lut3d`` filter. This module
is the still-image equivalent: a parser plus a NumPy trilinear interpolator,
so one frame gets the same class of look without spawning an encoder process.

A ``.cube`` file stores an ``N x N x N`` grid of output RGB samples with
**R varying fastest** (then G, then B) -- the order the Adobe Cube
specification defines. :func:`apply_cube` trilinearly interpolates an image
through that grid.

:func:`load_cube` also reads the two other shapes LUT packs ship in -- a HALD
CLUT PNG and a 1D tone-curve ``.cube`` -- normalising both into the same 3D
:class:`Cube`, so :func:`apply_cube` only ever sees one input shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Per-axis resolution a 1D tone curve is expanded to. See
#: :func:`_separable_cube_from_curves` for why it is not the curve's own length.
LUT_1D_TO_3D_SIZE = 33


class CubeError(ValueError):
    """A LUT file is malformed, unreadable, or an unsupported variant."""


@dataclass(frozen=True)
class Cube:
    """A parsed 3D LUT: ``table`` indexed ``[r, g, b] -> (3,)`` output in [0, 1].

    ``size`` is the per-axis resolution; ``domain_min``/``domain_max`` are the
    input range the grid spans (almost always 0..1, but honoured if a file
    declares otherwise).
    """

    size: int
    table: np.ndarray  # (size, size, size, 3) float32, indexed [r, g, b]
    domain_min: np.ndarray  # (3,) float32
    domain_max: np.ndarray  # (3,) float32


def _table_from_rows(rows, size: int) -> np.ndarray:
    """Fold ``size**3`` R-fastest samples into a table indexed ``[r, g, b]``.

    Flat index = ``r + g*size + b*size**2`` (the Adobe Cube order, and the same
    order a HALD CLUT's pixels run in), so reshaping to ``(b, g, r, 3)`` and
    transposing gives the ``[r, g, b]`` table :func:`apply_cube` indexes.
    """
    table = np.asarray(rows, dtype=np.float32).reshape(size, size, size, 3)
    return np.ascontiguousarray(table.transpose(2, 1, 0, 3))


def _load_hald_png(path: Path) -> Cube:
    """Parse a HALD CLUT PNG -- the grid-image LUT format many free film-simulation
    packs ship instead of (or alongside) a ``.cube``.

    A level-``n`` HALD is a square ``n**3`` px image holding a cube of size
    ``S = n**2``, so the pixel count is exactly ``S**3`` and ``S`` is its integer
    cube root. Geometry is fail-closed: a non-square image, a pixel count that is
    not a perfect cube, or ``S < 2`` raises rather than reading a truncated table.

    Requires Pillow (``pip install pycube-lut[hald]``). Nothing else in this
    module imports it, so a ``.cube``-only install needs NumPy alone.
    """
    try:
        from PIL import Image

        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
    except Exception as exc:
        raise CubeError(f"{path}: cannot read HALD CLUT PNG: {exc}") from exc

    height, width = arr.shape[:2]
    if height != width:
        raise CubeError(f"{path}: HALD CLUT must be square, got {width}x{height}")
    pixels = height * width
    size = round(pixels ** (1 / 3))
    if size**3 != pixels:
        raise CubeError(f"{path}: {pixels} pixels is not a perfect cube — not a HALD CLUT")
    if size < 2:
        raise CubeError(f"{path}: HALD CLUT cube size must be at least 2, got {size}")

    peak = 65535.0 if arr.dtype == np.uint16 else 255.0
    rows = arr.reshape(-1, 3).astype(np.float32) / peak
    return Cube(
        size=size,
        table=_table_from_rows(rows, size),
        domain_min=np.zeros(3, dtype=np.float32),
        domain_max=np.ones(3, dtype=np.float32),
    )


def _separable_cube_from_curves(curves: np.ndarray, domain) -> Cube:
    """Turn three per-channel 1D tone curves into an equivalent separable 3D cube:
    ``table[i, j, k] = (curve_r[i], curve_g[j], curve_b[k])``.

    Each curve is first resampled (``np.interp``) onto a fixed
    :data:`LUT_1D_TO_3D_SIZE` grid, because the 3D table costs the cube of the 1D
    length: a common 1024-entry tone curve would otherwise mean a 1024**3 (~10**9)
    entry table. A tone curve is smooth by nature, so a 33-point resample is
    visually lossless at a ~36k-entry cost.
    """
    size = LUT_1D_TO_3D_SIZE
    src = np.linspace(0.0, 1.0, curves.shape[0])
    dst = np.linspace(0.0, 1.0, size)
    resampled = np.stack([np.interp(dst, src, curves[:, c]) for c in range(3)], axis=1)
    table = np.empty((size, size, size, 3), dtype=np.float32)
    table[..., 0] = resampled[:, 0].reshape(size, 1, 1)
    table[..., 1] = resampled[:, 1].reshape(1, size, 1)
    table[..., 2] = resampled[:, 2].reshape(1, 1, size)
    return Cube(size=size, table=table, domain_min=domain[0], domain_max=domain[1])


def load_cube(path: str | Path) -> Cube:
    """Parse an external LUT into a 3D :class:`Cube`. Three intake formats:

    * an Adobe ``.cube`` 3D LUT (``LUT_3D_SIZE``) -- the reference path;
    * a **HALD CLUT PNG** (dispatched on the ``.png`` suffix, case-insensitive,
      before the text read a PNG would blow up on) -- see ``_load_hald_png``;
    * a **1D tone-curve ``.cube``** (``LUT_1D_SIZE``), expanded into the equivalent
      separable 3D cube at :data:`LUT_1D_TO_3D_SIZE`.

    Raises :class:`CubeError` -- always naming the path -- on a missing/unreadable
    file, a bad size directive, bad HALD geometry, or a data-row count that does
    not match the declared size. Failure is always closed, never a silently
    truncated table.
    """
    path = Path(path)
    if path.suffix.lower() == ".png":
        return _load_hald_png(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise CubeError(f"{path}: cannot read LUT: {exc}") from exc

    size: int | None = None
    size_1d: int | None = None
    domain_min = np.zeros(3, dtype=np.float32)
    domain_max = np.ones(3, dtype=np.float32)
    data: list[tuple[float, float, float]] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split(None, 1)[0].upper()
        if key in ("LUT_1D_SIZE", "LUT_3D_SIZE"):
            try:
                parsed = int(line.split()[1])
            except (IndexError, ValueError):
                raise CubeError(f"{path}: malformed {key} line: {line!r}") from None
            if parsed < 2:
                raise CubeError(f"{path}: {key} must be >= 2, got {parsed}")
            if key == "LUT_1D_SIZE":
                size_1d = parsed
            else:
                size = parsed
            continue
        if key in ("DOMAIN_MIN", "DOMAIN_MAX"):
            try:
                vals = [float(v) for v in line.split()[1:4]]
            except ValueError:
                raise CubeError(f"{path}: malformed {key} line: {line!r}") from None
            if len(vals) != 3:
                raise CubeError(f"{path}: {key} needs 3 values, got {line!r}")
            target = domain_min if key == "DOMAIN_MIN" else domain_max
            target[:] = vals
            continue
        if key in ("TITLE",):
            continue
        # A data row: three floats. Anything else is an unknown directive we skip.
        parts = line.split()
        if len(parts) == 3:
            try:
                data.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue

    if size is None and size_1d is None:
        raise CubeError(f"{path}: no LUT_3D_SIZE — not a 3D .cube file")
    if np.any(domain_max <= domain_min):
        raise CubeError(f"{path}: DOMAIN_MAX must exceed DOMAIN_MIN on every channel")

    if size is None:  # a 1D tone-curve file: one row per entry, three curves wide
        if len(data) != size_1d:
            raise CubeError(f"{path}: expected {size_1d} data rows for a 1D LUT, found {len(data)}")
        curves = np.asarray(data, dtype=np.float32)
        return _separable_cube_from_curves(curves, (domain_min, domain_max))

    if len(data) != size**3:
        raise CubeError(
            f"{path}: expected {size**3} data rows for a {size}^3 LUT, found {len(data)}"
        )
    table = _table_from_rows(data, size)
    return Cube(size=size, table=table, domain_min=domain_min, domain_max=domain_max)


def apply_cube(rgb: np.ndarray, cube: Cube, strength: float = 1.0) -> np.ndarray:
    """Trilinearly interpolate an ``HxWx3`` image through ``cube``.

    Accepts uint8 or uint16 input and returns the SAME dtype: a film emulation
    laid over a 16-bit archival master must not be the step that throws the
    master's headroom away. Pure NumPy, no ffmpeg. Deterministic.

    ``strength`` is the LUT opacity in ``[0, 1]`` (clamped): the full-strength
    interpolated output is blended toward the ORIGINAL input in normalised
    float, ``out*strength + input*(1-strength)``, before quantising back.
    ``1.0`` (the default) is byte-identical to the full-strength LUT; ``0.0`` is
    a passthrough (the input, bar rounding).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise CubeError(f"expected an HxWx3 image, got shape {rgb.shape}")
    strength = min(1.0, max(0.0, float(strength)))
    peak = 65535.0 if rgb.dtype == np.uint16 else 255.0

    norm = rgb.astype(np.float32) / peak
    span = cube.domain_max - cube.domain_min
    t = np.clip((norm - cube.domain_min) / span, 0.0, 1.0)
    coords = t * (cube.size - 1)

    lo = np.clip(np.floor(coords).astype(np.intp), 0, cube.size - 2)
    frac = coords - lo
    r0, g0, b0 = lo[..., 0], lo[..., 1], lo[..., 2]
    r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1
    fr = frac[..., 0:1]
    fg = frac[..., 1:2]
    fb = frac[..., 2:3]

    table = cube.table
    out = (
        table[r0, g0, b0] * ((1 - fr) * (1 - fg) * (1 - fb))
        + table[r1, g0, b0] * (fr * (1 - fg) * (1 - fb))
        + table[r0, g1, b0] * ((1 - fr) * fg * (1 - fb))
        + table[r0, g0, b1] * ((1 - fr) * (1 - fg) * fb)
        + table[r1, g1, b0] * (fr * fg * (1 - fb))
        + table[r1, g0, b1] * (fr * (1 - fg) * fb)
        + table[r0, g1, b1] * ((1 - fr) * fg * fb)
        + table[r1, g1, b1] * (fr * fg * fb)
    )
    blended = out * strength + norm * (1.0 - strength)
    return (np.clip(blended, 0.0, 1.0) * peak).round().astype(rgb.dtype)
