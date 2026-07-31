"""Scene-aware selection: ``characterize_cube`` measures what a cube DOES to the
image, ``score_lut`` scores that against what the scene wants AND against how much
of the image it kills, ``select_luts`` keeps the best N — deterministically."""

from dataclasses import replace

import numpy as np
import pytest

from pycube_lut import (
    LUT_MAX_CRUSH,
    SceneHints,
    characterize_cube,
    load_cube,
    select_luts,
)

#: name -> cube transform, including the two traps: ``bw``, which a signed "a rich
#: image wants less chroma" demand would hand every low-headroom scene, and ``crush``,
#: whose rolloff reads mild over a 0-1 sweep and eats a real image's whole shadow range.
LOOKS = {
    "neutral": lambda r, g, b: (r, g, b),
    "warm": lambda r, g, b: (min(1.0, r * 1.18), g, b * 0.82),
    "punch": lambda r, g, b: tuple(min(1.0, max(0.0, 0.4 + 1.3 * (v - 0.4))) for v in (r, g, b)),
    "bw": lambda r, g, b: ((r + g + b) / 3.0,) * 3,
    "crush": lambda r, g, b: tuple(max(0.0, (v - 0.3) / 0.7) for v in (r, g, b)),
}
#: A frame in the range a developed photo ACTUALLY occupies — not a 0-1 lattice.
_L = np.linspace(36.0, 168.0, 24)
FRAME = np.stack(np.meshgrid(_L, _L, _L, indexing="ij"), -1).reshape(1, -1, 3).astype(np.uint8)


def _cube(path, look="neutral", size=9):
    fn, axis = LOOKS[look], np.linspace(0.0, 1.0, size)
    rows = ["{:.6f} {:.6f} {:.6f}".format(*fn(r, g, b)) for b in axis for g in axis for r in axis]
    path.write_text("\n".join([f"LUT_3D_SIZE {size}", *rows]) + "\n")
    return load_cube(path)


def _library(tmp_path, names=tuple(LOOKS)):
    return {n: _cube(tmp_path / f"{n}.cube", n) for n in names}


def _scene(**kwargs):
    return replace(SceneHints(), **kwargs)


def _ranked(cubes, scene):
    return [n for n, _, _ in select_luts(list(cubes.items()), scene, FRAME, len(cubes))]


def test_cubes_characterise_on_the_axis_they_move(tmp_path):
    t = {n: characterize_cube(c, FRAME) for n, c in _library(tmp_path).items()}
    assert t["neutral"] == pytest.approx((0.0, 1.0, 1.0, 0.0), abs=1e-3)  # warm/contrast/sat/crush
    assert t["warm"].warm_shift > 0.03 > abs(t["punch"].warm_shift)
    assert t["punch"].contrast_gain > 1.15 > t["warm"].contrast_gain
    assert t["bw"].saturation_gain == pytest.approx(0.0, abs=1e-3)
    assert t["crush"].crushed > LUT_MAX_CRUSH  # measured where the image LIVES


def test_scoring_matches_the_cube_to_the_scene_deterministically(tmp_path):
    cubes = _library(tmp_path)
    cubes["alpha"] = _cube(tmp_path / "alpha.cube", "warm")  # scores tie with "warm"
    # An already-warm image must not cook. A rich one wants a chroma LEAN, not zero
    # chroma: scoring every axis as distance from a TARGET (never "less is better")
    # is what stops the one B&W cube in a library winning every low-headroom scene.
    warm_rich = _scene(warmth="warm", sat_profile="rich", chroma_headroom=0.15)
    assert _ranked(cubes, warm_rich)[0] == "neutral"
    # A flat muted image wants spread and chroma — but never bought by killing shadows.
    muted_flat = _scene(sat_profile="muted", dynamic_range="flat", chroma_headroom=0.95)
    assert _ranked(cubes, muted_flat)[0] == "punch"
    assert _ranked(cubes, muted_flat)[-1] == "crush"
    cool_hazy = _scene(warmth="cool", haze="hazy", dynamic_range="flat", chroma_headroom=0.8)
    winners = {_ranked(cubes, s)[0] for s in (warm_rich, muted_flat, cool_hazy)}
    assert len(winners) >= 2, f"collapsed onto one LUT: {winners}"
    ranking = _ranked(cubes, cool_hazy)
    assert ranking == _ranked(cubes, cool_hazy) == _ranked(cubes, cool_hazy)
    assert ranking.index("alpha") < ranking.index("warm"), "identical scores break by name"


def test_select_luts_returns_the_best_n_with_their_scores(tmp_path):
    cubes = _library(tmp_path)
    picks = select_luts(list(cubes.items()), _scene(dynamic_range="flat"), FRAME, 2)
    assert len(picks) == 2
    assert [p[0] for p in picks] == _ranked(cubes, _scene(dynamic_range="flat"))[:2]
    assert picks[0][2] >= picks[1][2]  # descending score, and the score is exposed


@pytest.mark.parametrize("count", [0, -1, 6])
def test_asking_for_an_impossible_count_raises(tmp_path, count):
    cubes = _library(tmp_path)  # 5 candidates
    with pytest.raises(ValueError, match="cannot pick"):
        select_luts(list(cubes.items()), SceneHints(), FRAME, count)


def test_any_object_with_the_scene_attributes_works(tmp_path):
    """The scene is a structural type — no dependency on this package's dataclass."""

    class MyAnalysis:
        exposure_class = "normal"
        warmth = "cool"
        sat_profile = "muted"
        haze = "hazy"
        dynamic_range = "flat"
        is_near_monochrome = False
        chroma_headroom = 0.8
        warm_fraction = 0.1

    cubes = _library(tmp_path)
    assert _ranked(cubes, MyAnalysis()) == _ranked(
        cubes,
        _scene(
            warmth="cool",
            sat_profile="muted",
            haze="hazy",
            dynamic_range="flat",
            chroma_headroom=0.8,
            warm_fraction=0.1,
        ),
    )
