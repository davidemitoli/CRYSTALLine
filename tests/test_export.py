"""Image and phonon-animation export (off-screen PyVista)."""

import os

import numpy as np
import pytest

pytest.importorskip("pyvista")
pytest.importorskip("ase")

import pyvista as pv  # noqa: E402
from ase.build import bulk  # noqa: E402

from crystalline.core.phonons import PhononMode  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402
from crystalline.viz import export  # noqa: E402
from crystalline.viz.render_settings import RenderSettings  # noqa: E402
from crystalline.viz.renderer import StructureRenderer  # noqa: E402


@pytest.fixture
def nacl():
    return Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))


def _plotter_with(structure):
    p = pv.Plotter(off_screen=True)
    StructureRenderer(p).set_structure(structure)
    p.reset_camera()
    return p


def test_save_view_image_raster_and_vector(nacl, tmp_path):
    p = _plotter_with(nacl)
    png = export.save_view_image(p, str(tmp_path / "view.png"))
    svg = export.save_view_image(p, str(tmp_path / "view.svg"))
    p.close()
    assert os.path.getsize(png) > 0
    assert os.path.getsize(svg) > 0


def test_save_view_image_rejects_unknown_format(nacl, tmp_path):
    p = _plotter_with(nacl)
    with pytest.raises(ValueError):
        export.save_view_image(p, str(tmp_path / "view.xyz"))
    p.close()


def test_render_animation_frames_shape(nacl):
    mode = PhononMode(120.0, np.array([[1, 0, 0], [-1, 0, 0]], float))
    frames = export.render_animation_frames(
        nacl, nacl.positions, mode, RenderSettings(), amplitude=0.6, n_frames=6
    )
    assert len(frames) == 6
    assert frames[0].ndim == 3 and frames[0].shape[2] in (3, 4)
    assert frames[0].dtype == np.uint8
    # the mode actually moves atoms, so not every frame is identical
    assert not all(np.array_equal(frames[0], f) for f in frames[1:])


def test_save_animation_gif(tmp_path):
    frames = [np.full((20, 30, 3), i * 20, np.uint8) for i in range(5)]
    out = export.save_animation(frames, str(tmp_path / "anim.gif"), fps=10)
    assert out == [str(tmp_path / "anim.gif")]
    from PIL import Image

    with Image.open(out[0]) as im:
        assert getattr(im, "n_frames", 1) == 5


def test_save_animation_png_sequence(tmp_path):
    frames = [np.zeros((10, 10, 3), np.uint8) for _ in range(4)]
    written = export.save_animation(frames, str(tmp_path / "frame.png"))
    assert len(written) == 4
    assert all(os.path.exists(p) for p in written)
    assert os.path.basename(written[0]) == "frame_000.png"


def test_save_animation_rejects_unknown_and_empty(tmp_path):
    with pytest.raises(ValueError):
        export.save_animation([], str(tmp_path / "x.gif"))
    with pytest.raises(ValueError):
        export.save_animation([np.zeros((4, 4, 3), np.uint8)], str(tmp_path / "x.qqq"))
