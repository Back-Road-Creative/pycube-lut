"""Pick the LUT that suits an image, out of a library of them.

A ``.cube`` is an opaque table: there is no metadata saying "this one warms",
so a LUT cannot be matched to an image by name or by reading the file. The only
way to know what it does is to **run it and measure the difference**.

That is what this module does, in three steps:

1. :func:`characterize_cube` applies a cube to a small view of the image and
   measures four things it changed -- warmth, contrast, saturation, and how much
   of the frame it destroyed.
2. :func:`score_lut` scores those measurements against what the image *wants*,
   described by a :class:`Scene` (use :class:`SceneHints`, or your own object
   with the same attributes).
3. :func:`select_luts` ranks a whole library and returns the best N.

**Measure on the real image, never on a 0-1 lattice.** A cube's shadow rolloff
lands wherever the image's own histogram lives. A developed photo typically
occupies roughly luma 0.14--0.66 and rarely clips, so a rolloff that looks mild
when swept over the full 0-1 range can eat the entire shadow range of an actual
frame. Feed :func:`characterize_cube` a downsampled copy of the image you will
ship, already developed -- not a synthetic ramp.

Everything here is pure and deterministic: the same library, image and scene
always produce the same ordered list, with ties broken by name.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

from .cube import Cube, apply_cube

#: Suggested long-edge size (px) for the metering view you pass in. Small enough
#: that scoring a large library is cheap, large enough to keep the histogram.
LUT_PROBE_DIM = 96

NEAR_BLACK = 16.0 / 255.0  # under this a pixel is crushed; over NEAR_WHITE it is blown
NEAR_WHITE = 249.0 / 255.0

# Every axis scores as DISTANCE FROM THE TARGET the scene wants, never "more of
# what it wants is better". A signed demand lets the most extreme cube on the
# demanded axis win outright -- in a large library that means one monochrome
# conversion takes every low-chroma scene, which is never the right answer.
WARM_COOK_WEIGHT = 10.0  # warmer than the scene wants — the worst error: it cooks
WARM_CHILL_WEIGHT = 6.0  # cooler than it wants: wrong, but recoverable
CONTRAST_WEIGHT, SATURATION_WEIGHT = 3.0, 2.0
COOL_SCENE_WARM_TARGET = 0.08  # only a measured-cool frame wants warmth added back
CONTRAST_TARGET_SPAN = 0.15  # what a demand of 1.0 asks for: ~15% more spread...
SATURATION_TARGET_SPAN = 0.15  # ...or ~15% more chroma. Deliberately modest.
LUT_MAX_CRUSH = 0.10  # frame share killed past which a candidate cannot win at all
CRUSH_WEIGHT = 8.0
CRUSH_SUPPRESSED = 100.0  # sinks a frame-killing candidate below every intact one
WARM_FRACTION_ALREADY_WARM = 0.5  # warm-hue chroma share at which a frame reads warm


class Scene(Protocol):
    """What :func:`score_lut` needs to know about the image.

    Structural only -- any object exposing these attributes works, so you can
    plug in whatever your own analysis pass already produces. :class:`SceneHints`
    is a ready-made implementation.
    """

    exposure_class: str  # low_key | normal | bright | overexposed
    warmth: str  # cool | neutral | warm
    sat_profile: str  # muted | normal | rich
    haze: str  # clear | hazy | foggy
    dynamic_range: str  # flat | normal | wide
    is_near_monochrome: bool
    chroma_headroom: float  # 0..1 — how much extra chroma the image can absorb
    warm_fraction: float  # 0..1 — chroma-weighted share of warm (red/orange/yellow) hues


@dataclass(frozen=True)
class SceneHints:
    """A plain :class:`Scene`. Every field has a neutral default, so you only set
    the ones your own analysis is confident about."""

    exposure_class: str = "normal"
    warmth: str = "neutral"
    sat_profile: str = "normal"
    haze: str = "clear"
    dynamic_range: str = "normal"
    is_near_monochrome: bool = False
    chroma_headroom: float = 0.5
    warm_fraction: float = 0.0


class LutTraits(NamedTuple):
    """Measured: mean (R-B) added (>0 warms), luma-spread gain, saturation gain, kill share."""

    warm_shift: float
    contrast_gain: float
    saturation_gain: float
    crushed: float


def _probe_stats(frame: np.ndarray) -> tuple[float, float, float, float]:
    """``(mean R-B, luma spread, mean saturation, dead share)`` of a uint8 view."""
    f = frame.astype(np.float32) / 255.0
    rgb_max = f.max(axis=-1)  # saturation on the (max - min) / max basis
    sat = np.where(rgb_max > 0, (rgb_max - f.min(axis=-1)) / np.maximum(rgb_max, 1e-6), 0.0)
    luma = f.mean(axis=-1)
    dead = float(np.mean((luma < NEAR_BLACK) | (luma > NEAR_WHITE)))
    return float((f[..., 0] - f[..., 2]).mean()), float(luma.std()), float(sat.mean()), dead


def characterize_cube(cube: Cube, frame: np.ndarray) -> LutTraits:
    """Measure what ``cube`` does to THIS frame -- pure and deterministic.

    ``frame`` is a uint8 ``HxWx3`` view of the image you intend to ship (see the
    module docstring on why it must not be a synthetic ramp). Not cached: the
    traits are a function of the frame, and one pass measures each once.
    """
    in_rb, in_spread, in_sat, in_dead = _probe_stats(frame)
    out_rb, out_spread, out_sat, out_dead = _probe_stats(apply_cube(frame, cube))
    return LutTraits(
        out_rb - in_rb,
        out_spread / in_spread if in_spread > 0.01 else 1.0,
        out_sat / in_sat if in_sat > 0.01 else 1.0,
        max(0.0, out_dead - in_dead),
    )


def score_lut(cube: Cube, scene: Scene, frame: np.ndarray) -> float:
    """How well ``cube`` suits ``scene`` on ``frame``: higher is better, deterministic."""
    traits = characterize_cube(cube, frame)
    # Warmth: only a cool frame wants warmth added; an already-warm one must not
    # cook under a warm LUT.
    warm_scene = scene.warmth == "warm" or scene.warm_fraction >= WARM_FRACTION_ALREADY_WARM
    delta = traits.warm_shift - (
        COOL_SCENE_WARM_TARGET if scene.warmth == "cool" and not warm_scene else 0.0
    )
    score = -(WARM_COOK_WEIGHT if delta > 0 else -WARM_CHILL_WEIGHT) * delta
    # Contrast: a flat or veiled frame wants spread; a wide-range or low-key one
    # is already dense and would lose its shadows to a contrasty cube.
    contrast_demand = 1.0 if (scene.dynamic_range == "flat" or scene.haze != "clear") else 0.0
    contrast_demand -= 0.6 if scene.dynamic_range == "wide" else 0.0
    contrast_demand -= 0.5 if scene.exposure_class == "low_key" else 0.0
    contrast_target = 1.0 + CONTRAST_TARGET_SPAN * contrast_demand
    score -= CONTRAST_WEIGHT * abs(traits.contrast_gain - contrast_target)
    # Saturation: chroma_headroom IS how much push the frame can absorb — a muted
    # frame takes a saturating LUT, a rich one a gentle desaturation (a LEAN, never
    # the floor), a near-monochrome one has no chroma to push at all.
    sat_demand = 2.0 * min(1.0, max(0.0, scene.chroma_headroom)) - 1.0
    sat_demand += 0.5 if scene.sat_profile == "muted" else 0.0
    sat_demand -= 0.5 if scene.sat_profile == "rich" else 0.0
    sat_demand = -1.5 if scene.is_near_monochrome else sat_demand
    sat_target = 1.0 + SATURATION_TARGET_SPAN * sat_demand
    score -= SATURATION_WEIGHT * abs(traits.saturation_gain - sat_target)
    # Crush: a rolloff landing inside THIS frame's histogram destroys the image, so past
    # LUT_MAX_CRUSH the candidate sinks below every intact one however well it reads on
    # taste; the linear term still ranks the least-bad first when they all crush.
    killed = CRUSH_SUPPRESSED if traits.crushed > LUT_MAX_CRUSH else 0.0
    return score - CRUSH_WEIGHT * traits.crushed - killed


def select_luts(
    candidates: Sequence[tuple[str, Cube]],
    scene: Scene,
    frame: np.ndarray,
    count: int,
) -> list[tuple[str, Cube, float]]:
    """Rank ``[(name, Cube), ...]`` for ``scene`` on ``frame``, best ``count`` first as
    ``[(name, cube, score), ...]``. Deterministic: :func:`score_lut` is pure and ties
    break by NAME, so one scene, frame and set always yield ONE ordered list."""
    if not 1 <= count <= len(candidates):
        raise ValueError(f"cannot pick {count} LUT(s) from {len(candidates)} candidate(s)")
    scored = [(name, cube, score_lut(cube, scene, frame)) for name, cube in candidates]
    scored.sort(key=lambda entry: (-entry[2], entry[0]))
    return scored[:count]
