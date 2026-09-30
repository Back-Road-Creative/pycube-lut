"""Reader + trilinear applier: parsing, indexing order, domain, formats, strength."""

import warnings

import numpy as np
import pytest

from pycube_lut import (
    LUT_1D_TO_3D_SIZE,
    Cube,
    CubeError,
    apply_cube,
    load_cube,
)


def _write_cube(path, size, fn) -> None:
    """Write a size^3 .cube whose grid sample at (r,g,b) is ``fn(r,g,b)`` — in the
    Adobe R-fastest order the reader must honor."""
    axis = np.linspace(0.0, 1.0, size)
    lines = [f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    for b in axis:
        for g in axis:
            for r in axis:
                o = fn(r, g, b)
                lines.append(f"{o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
    path.write_text("\n".join(lines) + "\n")


def test_load_cube_parses_size_domain_and_indexing(tmp_path):
    p = tmp_path / "id.cube"
    _write_cube(p, 3, lambda r, g, b: (r, g, b))
    cube = load_cube(p)
    assert cube.size == 3
    assert np.allclose(cube.domain_min, 0.0) and np.allclose(cube.domain_max, 1.0)
    # table is indexed [r, g, b]: the identity grid must read back as its coords.
    assert np.allclose(cube.table[2, 0, 0], [1.0, 0.0, 0.0])  # max R, no G/B
    assert np.allclose(cube.table[0, 2, 0], [0.0, 1.0, 0.0])  # max G


def test_identity_lut_is_a_near_noop(tmp_path):
    p = tmp_path / "id.cube"
    _write_cube(p, 33, lambda r, g, b: (r, g, b))
    cube = load_cube(p)
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
    out = apply_cube(img, cube)
    # Trilinear interpolation of an identity (affine) grid is exact bar rounding.
    assert np.abs(out.astype(int) - img).max() <= 1


def test_channel_swap_lut_reorders_channels(tmp_path):
    # A LUT that outputs (b, r, g) proves the applier honors per-channel indexing.
    p = tmp_path / "swap.cube"
    _write_cube(p, 33, lambda r, g, b: (b, r, g))
    cube = load_cube(p)
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[..., 0] = 200  # pure-ish red in
    out = apply_cube(img, cube)
    # red input -> the LUT routes R into the G output channel
    assert out[0, 0, 1] > 190 and out[0, 0, 0] < 10 and out[0, 0, 2] < 10


def _reference_grade(rgb: np.ndarray) -> np.ndarray:
    """An analytic grade: a gamma lift, then a saturation push about per-pixel luma.
    Float in and out, both in [0, 1]. Non-separable across channels on purpose —
    a per-channel curve would not exercise the 3D lookup."""
    v = np.clip(rgb, 0.0, 1.0) ** 1.15
    luma = v.mean(axis=-1, keepdims=True)
    return np.clip(luma + (v - luma) * 1.30, 0.0, 1.0)


def _bake_cube(path, fn, size) -> None:
    """Bake ``fn`` into a size^3 .cube by sampling it on the grid, R-fastest."""
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    rows = fn(grid)
    body = "\n".join(f"{o[0]:.6f} {o[1]:.6f} {o[2]:.6f}" for o in rows)
    path.write_text(f"LUT_3D_SIZE {size}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n{body}\n")


def test_baked_lut_matches_the_function_it_was_baked_from(tmp_path):
    # Round-trip parity: bake a known transform into a .cube, then read it back and
    # apply it. The result must reproduce the SAME grade the function computes
    # directly — the whole point of shipping a look as a table.
    cube_path = tmp_path / "baked.cube"
    _bake_cube(cube_path, _reference_grade, size=64)
    cube = load_cube(cube_path)

    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    via_lut = apply_cube(img, cube).astype(int)
    direct = (_reference_grade(img.astype(np.float32) / 255.0) * 255.0).round().astype(int)
    assert np.abs(via_lut - direct).mean() < 1.5  # cube quantization + trilinear only


def test_domain_scaling_is_honored(tmp_path):
    # A LUT declaring a 0..0.5 domain maps mid-grey to the grid's top.
    p = tmp_path / "dom.cube"
    axis = np.linspace(0.0, 1.0, 2)
    lines = ["LUT_3D_SIZE 2", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 0.5 0.5 0.5"]
    for b in axis:
        for g in axis:
            for r in axis:
                lines.append(f"{r:.6f} {g:.6f} {b:.6f}")
    p.write_text("\n".join(lines) + "\n")
    cube = load_cube(p)
    img = np.full((2, 2, 3), 128, dtype=np.uint8)  # ~0.5 in -> domain max -> grid 1.0
    out = apply_cube(img, cube)
    assert out.min() >= 250


def test_inverted_domain_raises(tmp_path):
    p = tmp_path / "baddom.cube"
    p.write_text("LUT_3D_SIZE 2\nDOMAIN_MIN 1 1 1\nDOMAIN_MAX 0 0 0\n" + "0 0 0\n" * 8)
    with pytest.raises(CubeError, match="DOMAIN_MAX must exceed DOMAIN_MIN"):
        load_cube(p)


def test_comments_and_title_are_skipped(tmp_path):
    p = tmp_path / "c.cube"
    body = "\n".join(f"{v:.3f} {v:.3f} {v:.3f}" for v in (0.0, 1.0) for _ in range(4))
    p.write_text(f'# a comment\nTITLE "Film Look"\nLUT_3D_SIZE 2\n\n{body}\n')
    cube = load_cube(p)
    assert cube.size == 2


def test_wrong_row_count_raises(tmp_path):
    p = tmp_path / "short.cube"
    p.write_text("LUT_3D_SIZE 2\n0 0 0\n1 1 1\n")  # needs 8 rows
    with pytest.raises(CubeError, match="expected 8 data rows"):
        load_cube(p)


def test_an_absurd_size_directive_fails_fast_without_allocating(tmp_path):
    # A hostile 25-byte file must not make the reader try to build a 10^15-entry
    # table: the row count is checked against the declared size before any array.
    p = tmp_path / "bomb.cube"
    p.write_text("LUT_3D_SIZE 100000\n0 0 0\n")
    with pytest.raises(CubeError, match="expected .* data rows"):
        load_cube(p)


def _write_hald(path, level, fn) -> None:
    """Write a level-``level`` HALD CLUT PNG (side ``level**3`` px, cube size
    ``level**2``) whose sample at (r,g,b) is ``fn(r,g,b)`` — R-fastest row-major,
    the same order a .cube's data rows use. ``fn`` takes/returns float arrays."""
    from PIL import Image

    size, side = level**2, level**3
    idx = np.arange(side * side)
    coords = (
        (idx % size) / (size - 1),
        ((idx // size) % size) / (size - 1),
        (idx // size**2) / (size - 1),
    )
    out = np.stack(fn(*coords), axis=1)
    arr = (out * 255).round().astype(np.uint8).reshape(side, side, 3)
    Image.fromarray(arr).save(path)


def test_hald_identity_round_trips(tmp_path):
    p = tmp_path / "id.png"
    _write_hald(p, 2, lambda r, g, b: (r, g, b))  # level 2 -> 8x8 px, 4^3 cube
    cube = load_cube(p)
    assert cube.size == 4
    assert np.allclose(cube.domain_min, 0.0) and np.allclose(cube.domain_max, 1.0)
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    assert np.abs(apply_cube(img, cube).astype(int) - img).max() <= 1


def test_hald_channel_swap_reorders_channels(tmp_path):
    p = tmp_path / "swap.png"
    _write_hald(p, 3, lambda r, g, b: (b, r, g))  # level 3 -> 27x27 px, 9^3 cube
    cube = load_cube(p)
    assert cube.size == 9
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[..., 0] = 200  # red in -> the LUT routes R into the G output channel
    out = apply_cube(img, cube)
    assert out[0, 0, 1] > 190 and out[0, 0, 0] < 10 and out[0, 0, 2] < 10


@pytest.mark.parametrize(
    "shape, match",
    [((8, 16), "square"), ((5, 5), "perfect cube"), ((1, 1), "at least 2")],
)
def test_hald_bad_geometry_raises(tmp_path, shape, match):
    from PIL import Image

    p = tmp_path / "bad.png"
    Image.fromarray(np.zeros((*shape, 3), np.uint8)).save(p)
    with pytest.raises(CubeError, match=match) as exc:
        load_cube(p)
    assert str(p) in str(exc.value)


def test_unreadable_hald_png_raises(tmp_path):
    p = tmp_path / "junk.PNG"  # suffix dispatch is case-insensitive
    p.write_bytes(b"not a png at all")
    with pytest.raises(CubeError, match="cannot read HALD CLUT PNG") as exc:
        load_cube(p)
    assert str(p) in str(exc.value)


def test_one_d_lut_becomes_a_separable_3d_cube(tmp_path):
    # Three per-channel tone curves: R inverted, G and B identity.
    n = 16
    axis = np.linspace(0.0, 1.0, n)
    p = tmp_path / "curve.cube"
    p.write_text(f"LUT_1D_SIZE {n}\n" + "".join(f"{1 - v:.6f} {v:.6f} {v:.6f}\n" for v in axis))
    cube = load_cube(p)
    assert cube.size == LUT_1D_TO_3D_SIZE  # resampled, never 1024^3
    out = apply_cube(np.full((2, 2, 3), 64, np.uint8), cube)
    assert abs(int(out[0, 0, 0]) - 191) <= 2  # R inverted
    assert abs(int(out[0, 0, 1]) - 64) <= 2  # G untouched
    assert abs(int(out[0, 0, 2]) - 64) <= 2  # B untouched


@pytest.mark.parametrize(
    "text, match",
    [
        ("LUT_1D_SIZE 4\n0 0 0\n1 1 1\n", "expected 4 data rows"),
        ("LUT_1D_SIZE x\n0 0 0\n", "malformed LUT_1D_SIZE"),
        ("LUT_1D_SIZE 1\n0 0 0\n", "must be >= 2"),
    ],
)
def test_one_d_lut_failures_raise(tmp_path, text, match):
    p = tmp_path / "bad1d.cube"
    p.write_text(text)
    with pytest.raises(CubeError, match=match) as exc:
        load_cube(p)
    assert str(p) in str(exc.value)


def test_three_d_cube_parses_to_an_rgb_indexed_identity_grid(tmp_path):
    p = tmp_path / "id5.cube"
    _write_cube(p, 5, lambda r, g, b: (r, g, b))
    cube = load_cube(p)
    axis = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    expected = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).astype(np.float32)
    assert cube.size == 5
    assert np.array_equal(cube.table, expected)
    assert np.array_equal(cube.domain_min, np.zeros(3, np.float32))
    assert np.array_equal(cube.domain_max, np.ones(3, np.float32))


def test_missing_size_raises(tmp_path):
    p = tmp_path / "nosize.cube"
    p.write_text("0 0 0\n1 1 1\n")
    with pytest.raises(CubeError, match="not a 3D .cube"):
        load_cube(p)


def test_unreadable_file_raises():
    with pytest.raises(CubeError, match="cannot read LUT"):
        load_cube("/no/such/film.cube")


def test_apply_requires_hwc3():
    cube = Cube(
        2, np.zeros((2, 2, 2, 3), np.float32), np.zeros(3, np.float32), np.ones(3, np.float32)
    )
    with pytest.raises(CubeError, match="HxWx3"):
        apply_cube(np.zeros((4, 4), np.uint8), cube)


def test_uint16_input_keeps_its_headroom(tmp_path):
    # A 16-bit master must come back 16-bit: quantizing it to 8 bits here would
    # silently throw away the headroom the master exists to carry.
    p = tmp_path / "id.cube"
    _write_cube(p, 33, lambda r, g, b: (r, g, b))
    cube = load_cube(p)
    img = np.linspace(0, 65535, 4 * 4 * 3).reshape(4, 4, 3).astype(np.uint16)
    out = apply_cube(img, cube)
    assert out.dtype == np.uint16
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 2


def test_default_strength_matches_full_strength_byte_for_byte(tmp_path):
    # strength defaults to 1.0 — the full-LUT output, unchanged bit for bit.
    p = tmp_path / "swap.cube"
    _write_cube(p, 33, lambda r, g, b: (b, r, g))
    cube = load_cube(p)
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    assert np.array_equal(apply_cube(img, cube), apply_cube(img, cube, 1.0))


def test_zero_strength_is_a_passthrough_to_the_input(tmp_path):
    # A channel-swap LUT visibly re-grades at full strength; at 0.0 it is a no-op.
    p = tmp_path / "swap.cube"
    _write_cube(p, 33, lambda r, g, b: (b, r, g))
    cube = load_cube(p)
    rng = np.random.default_rng(6)
    img = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    assert np.abs(apply_cube(img, cube, 1.0).astype(int) - img).mean() > 20  # full moves pixels
    assert np.abs(apply_cube(img, cube, 0.0).astype(int) - img).max() <= 1  # input, bar rounding


def test_intermediate_strength_lands_between_input_and_full_lut(tmp_path):
    # A LUT that crushes everything to black: half strength is halfway to black.
    p = tmp_path / "black.cube"
    _write_cube(p, 2, lambda r, g, b: (0.0, 0.0, 0.0))
    cube = load_cube(p)
    img = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert apply_cube(img, cube, 1.0).max() <= 1  # full LUT -> black
    assert np.abs(apply_cube(img, cube, 0.5).astype(int) - 100).max() <= 1  # midpoint of 200 and 0


def test_out_of_range_strength_is_clamped(tmp_path):
    p = tmp_path / "black.cube"
    _write_cube(p, 2, lambda r, g, b: (0.0, 0.0, 0.0))
    cube = load_cube(p)
    img = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert np.array_equal(apply_cube(img, cube, 5.0), apply_cube(img, cube, 1.0))
    assert np.array_equal(apply_cube(img, cube, -2.0), apply_cube(img, cube, 0.0))


# --- LUT-1: non-finite / unrepresentable values are rejected at the boundary ----------


def _cube_text(header: str, rows: list[str]) -> str:
    return header + "\n" + "\n".join(rows) + "\n"


@pytest.mark.parametrize("bad", ["nan", "NaN", "inf", "-inf", "infinity", "1e999", "1e39"])
def test_non_finite_table_entry_raises_before_apply(tmp_path, bad):
    # 1e39 is finite as a Python float but overflows the float32 table to inf.
    p = tmp_path / "badrow.cube"
    rows = ["0 0 0"] * 7 + [f"{bad} 0.5 0.5"]
    p.write_text(_cube_text("LUT_3D_SIZE 2", rows))
    with pytest.raises(CubeError, match="finite") as exc:
        load_cube(p)
    assert str(p) in str(exc.value)


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "1e39"])
@pytest.mark.parametrize("key", ["DOMAIN_MIN", "DOMAIN_MAX"])
def test_non_finite_domain_raises(tmp_path, key, bad):
    p = tmp_path / "baddom.cube"
    vals = f"{bad} 0 0" if key == "DOMAIN_MIN" else f"1 {bad} 1"
    p.write_text(_cube_text(f"LUT_3D_SIZE 2\n{key} {vals}", ["0 0 0"] * 8))
    with pytest.raises(CubeError, match="finite") as exc:
        load_cube(p)
    assert str(p) in str(exc.value)


@pytest.mark.parametrize("bad", ["nan", "inf", "1e39"])
def test_non_finite_1d_entry_raises(tmp_path, bad):
    p = tmp_path / "bad1d.cube"
    p.write_text(_cube_text("LUT_1D_SIZE 4", ["0 0 0", f"0.3 {bad} 0.3", "0.6 0.6 0.6", "1 1 1"]))
    with pytest.raises(CubeError, match="finite") as exc:
        load_cube(p)
    assert str(p) in str(exc.value)


def test_directly_constructed_cube_rejects_non_finite_values():
    table = np.zeros((2, 2, 2, 3), np.float32)
    table[1, 1, 1, 2] = np.nan
    with pytest.raises(CubeError, match="finite"):
        Cube(2, table, np.zeros(3, np.float32), np.ones(3, np.float32))
    with pytest.raises(CubeError, match="finite"):
        Cube(
            2,
            np.zeros((2, 2, 2, 3), np.float32),
            np.array([0, np.nan, 0], np.float32),
            np.ones(3, np.float32),
        )


def test_directly_constructed_cube_rejects_inverted_domain():
    with pytest.raises(CubeError, match="DOMAIN_MAX must exceed DOMAIN_MIN"):
        Cube(
            2,
            np.zeros((2, 2, 2, 3), np.float32),
            np.ones(3, np.float32),
            np.zeros(3, np.float32),
        )


def test_finite_out_of_unit_range_table_values_are_kept(tmp_path):
    # Legitimate LUTs overshoot [0, 1]; only non-finite values are rejected, and the
    # applier clamps on output without a cast warning.
    p = tmp_path / "overshoot.cube"
    rows = ["-0.25 1.5 0.5"] * 8
    p.write_text(_cube_text("LUT_3D_SIZE 2", rows))
    cube = load_cube(p)
    assert cube.table.min() == np.float32(-0.25) and cube.table.max() == np.float32(1.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = apply_cube(np.full((2, 2, 3), 100, np.uint8), cube)
    assert out.tolist() == [[[0, 255, 128]] * 2] * 2


def test_valid_fixtures_apply_without_runtime_warnings(tmp_path):
    p3 = tmp_path / "swap.cube"
    _write_cube(p3, 9, lambda r, g, b: (b, r, g))
    p1 = tmp_path / "curve.cube"
    p1.write_text(
        _cube_text("LUT_1D_SIZE 4", [f"{v:.6f} {v:.6f} {1 - v:.6f}" for v in np.linspace(0, 1, 4)])
    )
    img = np.full((3, 3, 3), 64, np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out3 = apply_cube(img, load_cube(p3))
        out1 = apply_cube(img, load_cube(p1))
    assert out3[0, 0].tolist() == [64, 64, 64]
    assert abs(int(out1[0, 0, 2]) - 191) <= 2


# --- LUT-2: only uint8 / uint16 pixels are accepted; nothing is silently 255-scaled ---


def _identity_cube(tmp_path):
    p = tmp_path / "id.cube"
    _write_cube(p, 5, lambda r, g, b: (r, g, b))
    return load_cube(p)


@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint32, np.uint64, np.bool_],
)
def test_unsupported_pixel_dtype_is_rejected_not_scaled(tmp_path, dtype):
    # A float32 [0, 1] image used to be divided by 255 and come back near-black (or,
    # cast back to float, as 0..255 values); signed ints wrapped or clipped. Refuse.
    cube = _identity_cube(tmp_path)
    img = np.full((2, 2, 3), 1, dtype=dtype)
    with pytest.raises(CubeError, match="uint8 or uint16") as exc:
        apply_cube(img, cube)
    assert np.dtype(dtype).name in str(exc.value)


def test_float_unit_range_image_cannot_silently_use_255_scaling(tmp_path):
    cube = _identity_cube(tmp_path)
    img = np.linspace(0.0, 1.0, 2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    with pytest.raises(CubeError, match="uint8 or uint16"):
        apply_cube(img, cube)


def test_uint16_precision_is_kept_beyond_8_bits(tmp_path):
    # A LUT that halves every channel: 40000 -> 20000. Quantizing through 8 bits
    # first would land on a multiple of ~128, so this pins real 16-bit precision.
    p = tmp_path / "half.cube"
    _write_cube(p, 2, lambda r, g, b: (r / 2, g / 2, b / 2))
    cube = load_cube(p)
    img = np.full((2, 2, 3), 40001, dtype=np.uint16)
    out = apply_cube(img, cube)
    assert out.dtype == np.uint16
    assert np.abs(out.astype(int) - 20000).max() <= 1


def test_uint8_known_output_is_unchanged(tmp_path):
    p = tmp_path / "half.cube"
    _write_cube(p, 2, lambda r, g, b: (r / 2, g / 2, b / 2))
    cube = load_cube(p)
    out = apply_cube(np.full((2, 2, 3), 200, dtype=np.uint8), cube)
    assert out.dtype == np.uint8
    assert np.abs(out.astype(int) - 100).max() <= 1
