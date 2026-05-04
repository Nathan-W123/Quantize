"""Offline parsers for cheap-opt xyz output (no ORCA binary needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.orca_cheap_opt import parse_xyz_trajectory_last


def test_parse_xyz_last_frame(tmp_path: Path):
    p = tmp_path / "two_frames.xyz"
    p.write_text(
        "2\nframe0\n"
        "O  0.0 0.0 0.1\n"
        "H  0.9 0.0 0.0\n"
        "2\nframe1\n"
        "O  0.0 0.0 0.0\n"
        "H  1.0 0.0 0.0\n",
        encoding="utf-8",
    )
    elems, coords = parse_xyz_trajectory_last(p)
    assert elems == ["O", "H"]
    np.testing.assert_allclose(coords[0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(coords[1], [1.0, 0.0, 0.0], atol=1e-12)
