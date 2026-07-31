# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0

First release.

### Added

- `load_cube()` — parse a LUT into a `Cube` (an `[r, g, b]`-indexed float32 table plus
  its input domain). Three intake formats normalised into one shape:
  - Adobe `.cube` 3D LUTs (`LUT_3D_SIZE`), R-varying-fastest data order, with
    `DOMAIN_MIN` / `DOMAIN_MAX`, `TITLE` and comments honoured;
  - HALD CLUT PNGs, dispatched on the `.png` suffix, with fail-closed geometry checks
    (square, perfect-cube pixel count, size ≥ 2);
  - 1D tone-curve `.cube` files (`LUT_1D_SIZE`), expanded into the equivalent separable
    3D cube resampled to 33 points per axis.
- `apply_cube()` — trilinear interpolation of an `HxWx3` uint8 or uint16 image through a
  `Cube`, returning the same dtype, with a `strength` opacity parameter in `[0, 1]`.
- `CubeError` — a `ValueError` subclass raised for every malformed or unreadable LUT.
  The message always names the path; a bad file never yields a truncated table.
- Optional scene-aware selection for choosing from a library of LUTs:
  `characterize_cube()` measures what a cube does to a specific image,
  `score_lut()` scores that against a `Scene`, and `select_luts()` returns the best N
  deterministically. `SceneHints` is a ready-made `Scene`; the protocol is structural,
  so any object with the same attributes works.
- Pillow is an optional extra (`pip install "pycube-lut[hald]"`) needed only by the
  HALD CLUT PNG reader; the core install is NumPy-only, and CI proves it.
