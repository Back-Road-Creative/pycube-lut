"""Pillow is an optional extra: everything but the HALD PNG path must work without it.

Run in a subprocess with Pillow blocked at import time, because this process has it
installed (the test suite writes HALD fixtures with it).
"""

import subprocess
import sys
import textwrap

_BLOCK_PILLOW = """
import sys

class _NoPillow:
    def find_spec(self, name, path=None, target=None):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("pillow is not installed")
        return None

sys.meta_path.insert(0, _NoPillow())
"""


def _run_without_pillow(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_PILLOW + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_core_import_and_apply_need_numpy_only(tmp_path):
    cube_path = tmp_path / "id.cube"
    axis = [0.0, 1.0]
    rows = "\n".join(f"{r} {g} {b}" for b in axis for g in axis for r in axis)
    cube_path.write_text(f"LUT_3D_SIZE 2\n{rows}\n")

    result = _run_without_pillow(f"""
        import numpy as np
        from pycube_lut import apply_cube, load_cube

        cube = load_cube({str(cube_path)!r})
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        out = apply_cube(img, cube)
        assert abs(int(out[0, 0, 0]) - 128) <= 1, out[0, 0]
        print("ok")
    """)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_hald_path_reports_the_missing_extra_as_a_cube_error(tmp_path):
    png = tmp_path / "look.png"
    png.write_bytes(b"")

    result = _run_without_pillow(f"""
        from pycube_lut import CubeError, load_cube

        try:
            load_cube({str(png)!r})
        except CubeError as exc:
            assert "cannot read HALD CLUT PNG" in str(exc), exc
            print("ok")
    """)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
