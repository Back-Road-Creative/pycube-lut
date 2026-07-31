# Contributing to pycube-lut

Thanks for your interest. This is a small, deliberately narrow library — a LUT reader
and a NumPy applier. Contributions that keep it small are the easiest to merge.

## Reporting

- **Bugs:** open an issue. For a parsing bug, attach the smallest LUT file that
  reproduces it (a 2×2×2 cube is usually enough) and say which tool wrote it.
- **Security vulnerabilities:** do **not** open a public issue — follow
  [SECURITY.md](SECURITY.md).

## Development setup

Python 3.11+ (the CI floor).

```bash
python -m venv .venv
.venv/bin/pip install -e ".[test]"
```

## The gates (run before you push)

CI runs these on every push and pull request:

```bash
.venv/bin/python -m pytest -q   # the suite
ruff check .                    # lint
ruff format --check .           # formatting
```

CI also installs the package *without* the `hald` extra and asserts the core import
still works. If you add a runtime dependency, that job is where it will fail — which is
the point: NumPy is the only hard dependency, and Pillow is needed by the HALD PNG
reader alone.

## Tests first

For a behaviour change, add or extend a test that fails before your change and passes
after. Never weaken or delete a test to make the suite pass.

Two things the tests deliberately pin, because they are the classic ways to get a LUT
wrong — please keep them green rather than adjusting them:

- **Data-row order is R-fastest**, then G, then B. `test_load_cube_parses_size_domain_and_indexing`
  and `test_three_d_cube_parses_to_an_rgb_indexed_identity_grid` hold that down.
- **Selection is measured on a real image, not a 0-1 lattice.** The `crush` fixture in
  `tests/test_select.py` exists because a rolloff that reads as mild over a synthetic
  ramp can destroy a real photograph's entire shadow range.

## Scope

Happily in scope: more LUT intake formats, correctness fixes, performance, tetrahedral
interpolation as an option, tiling for very large images.

Probably out of scope: colour-space management, a LUT *writer*, image I/O, a CLI. Those
belong in the layer above this one.

## Commits and pull requests

- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, …).
- One logical change per pull request, with the gate output in the description.
- Add a `## <version>` entry to [CHANGELOG.md](CHANGELOG.md) for anything user-visible.
- Update the README in the same change as the behaviour it documents.

## Conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
