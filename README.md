# pycube-lut

Load any `.cube` 3D LUT, HALD CLUT PNG, or 1D tone curve, and apply it to a NumPy
image in one call. Pure NumPy — no ffmpeg process, no OpenColorIO, no GPU.

```python
import numpy as np
from pycube_lut import apply_cube, load_cube

cube = load_cube("some_film_look.cube")     # or "look.png", or a 1D curve .cube
graded = apply_cube(image, cube)            # HxWx3 uint8 or uint16, same dtype back
half   = apply_cube(image, cube, 0.5)       # 50% opacity
```

## Why

A 3D lookup table is the only honest way to carry a film-style colour transform.
Global knobs — saturation, an S-curve, white balance, gamma, a black point — move
every hue the same way. A real stock's signature is hue-selective: it does one thing
to skin and a different thing to foliage. That needs a full three-dimensional lookup,
and a `.cube` file is how the world ships one.

Video tools apply these through ffmpeg's `lut3d` filter. If you have a single still
image already in memory, spawning an encoder to grade it is absurd — you write the
frame to disk, shell out, and read it back. This library is the still-image
equivalent: about 200 lines of NumPy that parse the table and trilinearly interpolate
your pixels through it.

It also absorbs the two other formats LUT packs ship in, so you do not have to care
which one you downloaded:

| You have | `load_cube` gives you |
| --- | --- |
| Adobe `.cube` with `LUT_3D_SIZE` | the table as-is |
| HALD CLUT PNG (a square grid image) | the same table, read from pixels |
| `.cube` with `LUT_1D_SIZE` (three tone curves) | an equivalent separable 3D cube |

One input shape reaches `apply_cube`, always.

## Install

```bash
pip install pycube-lut            # .cube and 1D curves — NumPy only
pip install "pycube-lut[hald]"    # adds Pillow, for reading HALD CLUT PNGs
```

Python 3.11+.

## Usage

### Grade an image

```python
import numpy as np
from PIL import Image                       # any HxWx3 uint8 array will do
from pycube_lut import apply_cube, load_cube

image = np.asarray(Image.open("photo.jpg").convert("RGB"))  # HxWx3 uint8
cube = load_cube("teal_orange.cube")

full  = apply_cube(image, cube)            # strength defaults to 1.0
subtle = apply_cube(image, cube, 0.35)     # blended back toward the original
```

`strength` is opacity in `[0, 1]` (values outside are clamped). The blend happens in
normalised float before quantising back, so `0.0` is a passthrough and `1.0` is
byte-identical to the full-strength LUT.

**uint16 in, uint16 out.** A look laid over a 16-bit master must not be the step that
throws the master's headroom away, so the output dtype always matches the input.

### Runnable end-to-end example

No LUT file handy? This writes one, then applies it — copy-paste and run:

```python
import numpy as np
from pycube_lut import apply_cube, load_cube

# A tiny .cube that swaps channels: R -> G, G -> B, B -> R.
size = 5
axis = np.linspace(0.0, 1.0, size)
rows = [f"{b:.4f} {r:.4f} {g:.4f}" for b in axis for g in axis for r in axis]
open("swap.cube", "w").write("\n".join([f"LUT_3D_SIZE {size}", *rows]) + "\n")

cube = load_cube("swap.cube")
print(cube.size, cube.table.shape)         # 5 (5, 5, 5, 3)

red = np.zeros((2, 2, 3), np.uint8)
red[..., 0] = 200
print(apply_cube(red, cube)[0, 0])         # [  0 200   0] — red became green
print(apply_cube(red, cube, 0.5)[0, 0])    # [100 100   0] — halfway there
```

Note the data-row order: **R varies fastest**, then G, then B. That is what the Adobe
Cube specification requires, and it is the single most common way to get a LUT wrong.

### Errors

Every failure raises `CubeError` (a `ValueError` subclass) and always names the path.
Failure is closed — a bad file never yields a silently truncated table.

```python
from pycube_lut import CubeError, load_cube

try:
    cube = load_cube(path)
except CubeError as exc:
    print(f"unusable LUT: {exc}")
```

### Picking a LUT automatically (optional)

If you have a *library* of LUTs and want the one that suits a given image, there is
no metadata to go on — a `.cube` is an opaque table, and the filename lies. The only
way to know what a LUT does is to run it and measure the difference:

```python
from pycube_lut import SceneHints, select_luts

library = [("sunset_warm", cube_a), ("cool_matte", cube_b), ("punchy", cube_c)]
scene = SceneHints(warmth="cool", dynamic_range="flat", chroma_headroom=0.8)

for name, cube, score in select_luts(library, scene, thumbnail, count=2):
    print(f"{name}: {score:.2f}")
```

`thumbnail` is a small uint8 copy of the image you actually intend to ship (96 px on
the long edge is plenty — see `LUT_PROBE_DIM`). **Do not pass a synthetic 0-1 ramp.**
A cube's shadow rolloff lands wherever your image's histogram lives; a rolloff that
looks mild when swept over the full range can eat the entire shadow range of a real
photograph, which typically occupies only about luma 0.14–0.66.

Scoring is deterministic and every axis is scored as *distance from what the scene
wants*, never "more is better" — otherwise the single most extreme LUT in a library
wins every time (a monochrome conversion takes every low-chroma image, which is never
what you meant). A candidate that destroys more than 10% of the frame is suppressed
below every intact one regardless of how good its colour reads.

`SceneHints` is a plain dataclass with neutral defaults, but `Scene` is a structural
type: any object exposing the same attributes works, so you can pass whatever your own
image-analysis pass already produces.

## Honest limits

- **Trilinear interpolation only.** Resolve, Photoshop and ffmpeg's `lut3d` default to
  *tetrahedral*, which differs slightly along the cube's diagonals. Expect small
  deviations from those tools on steep LUTs; use a denser LUT if it matters.
- **8-bit and 16-bit integer images only.** Float images are not handled; normalise to
  uint16 first.
- **Read-only.** There is no LUT *writer*, no `.cube` generator, and no LUT inversion.
- **1D tone curves are resampled to 33 points per axis.** The equivalent 3D table costs
  the cube of the 1D length, so a 1024-entry curve would mean ~10⁹ entries. Tone curves
  are smooth, so this is visually lossless — but it is a resample, not the original.
- **No colour management.** The LUT is applied to whatever encoding your pixels are
  already in. A LUT authored for log footage needs log input; feeding it sRGB will look
  wrong, and nothing here will warn you.
- **Whole image in memory.** The applier allocates several float32 arrays the size of
  the image. Tile very large images yourself.
- **HALD CLUT reading needs Pillow** (`pip install "pycube-lut[hald]"`). Nothing else
  does.
- **The selection heuristics are opinionated.** The weights in `pycube_lut.select` were
  tuned on landscape and travel photography. They are module constants — read them,
  and change them if your material is different.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
