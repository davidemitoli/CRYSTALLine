"""Export the 3D view: still images and phonon-mode animations.

* **Stills** are captured from a live plotter (WYSIWYG) — raster via
  ``screenshot`` (PNG/JPEG/…) or vector via ``save_graphic`` (SVG/PDF/EPS/…).
* **Animations** are rendered on a *fresh off-screen* plotter so the live view
  isn't disturbed and the whole thing stays testable headlessly. The same
  :class:`StructureRenderer` + :class:`PhononAnimator` are driven over one full
  vibration cycle; frames are written as an animated GIF (via Pillow, no extra
  deps), an MP4/video (via imageio-ffmpeg, if installed), or a PNG sequence.

Qt-free on purpose: the UI hands in the structure/mode/settings and a file path.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pyvista as pv

from crystalline.core.phonons import PhononMode
from crystalline.viz.phonon_animator import PhononAnimator
from crystalline.viz.renderer import StructureRenderer
from crystalline.viz.render_settings import RenderSettings

# Formats ``pyvista`` writes as a raster screenshot vs. a vector graphic.
RASTER_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
VECTOR_IMAGE_EXTS = (".svg", ".eps", ".ps", ".pdf", ".tex")
# Animation containers.
GIF_EXTS = (".gif",)
MOVIE_EXTS = (".mp4", ".mov", ".avi", ".webm", ".mkv")

# Defaults for an exported loop: smooth enough, small enough.
DEFAULT_FRAMES = 36
DEFAULT_FPS = 18


# ── still image ─────────────────────────────────────────────────────────────
def save_view_image(
    plotter, path: str, *, scale: int = 1, transparent: bool = False
) -> str:
    """Save the plotter's current view to ``path`` (raster or vector by extension).

    ``scale`` supersamples a raster screenshot (2 → twice the pixels each way,
    the 3D equivalent of a higher DPI); ``transparent`` drops the background so
    the structure sits on an alpha channel. Both apply to raster formats only —
    vector output (``save_graphic``) has neither knob and ignores them.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in VECTOR_IMAGE_EXTS:
        plotter.save_graphic(path)
    elif ext in RASTER_IMAGE_EXTS:
        plotter.screenshot(path, transparent_background=transparent, scale=max(1, int(scale)))
    else:
        raise ValueError(
            f"unsupported image format '{ext}'. Use one of "
            f"{', '.join(RASTER_IMAGE_EXTS + VECTOR_IMAGE_EXTS)}."
        )
    return path


# ── animation ───────────────────────────────────────────────────────────────
def render_animation_frames(
    structure,
    equilibrium: np.ndarray,
    mode: PhononMode,
    settings: Optional[RenderSettings] = None,
    *,
    amplitude: float = 0.5,
    n_frames: int = DEFAULT_FRAMES,
    reference_cell=None,
    bond_structure=None,
    camera=None,
    window_size=(800, 600),
) -> List[np.ndarray]:
    """Render one full vibration cycle off-screen; return a list of RGB frames.

    A fresh off-screen plotter/renderer is used so nothing about the live view
    changes. ``camera`` (a pyvista ``camera_position``) matches the on-screen
    orientation when supplied; otherwise the structure is auto-framed.
    """
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    try:
        renderer = StructureRenderer(plotter, settings)
        renderer.set_reference_cell(reference_cell)
        renderer.set_structure(structure, bond_structure=bond_structure)
        if camera is not None:
            plotter.camera_position = camera
        else:
            plotter.reset_camera()

        animator = PhononAnimator(renderer)
        animator.amplitude = amplitude
        animator.set_mode(np.asarray(equilibrium, dtype=float), mode)

        frames = []
        for phase in PhononAnimator.phase_sequence(n_frames):
            animator.set_frame(phase)
            frames.append(np.asarray(plotter.screenshot(return_img=True)))
        return frames
    finally:
        plotter.close()


def save_animation(frames: List[np.ndarray], path: str, *, fps: int = DEFAULT_FPS) -> List[str]:
    """Write ``frames`` to ``path``; format chosen by extension.

    ``.gif`` → animated GIF (Pillow); a video extension → MP4/… (imageio-ffmpeg);
    a raster image extension → a numbered PNG-style sequence (``stem_000.ext`` …).
    Returns the list of files written.
    """
    if not frames:
        raise ValueError("no frames to write")
    ext = os.path.splitext(path)[1].lower()
    if ext in GIF_EXTS:
        _save_gif(frames, path, fps)
        return [path]
    if ext in MOVIE_EXTS:
        _save_movie(frames, path, fps)
        return [path]
    if ext in RASTER_IMAGE_EXTS:
        return _save_frame_sequence(frames, path)
    raise ValueError(
        f"unsupported animation format '{ext}'. Use .gif, a video "
        f"({', '.join(MOVIE_EXTS)}), or an image extension for a frame sequence."
    )


def _save_gif(frames, path, fps) -> None:
    from PIL import Image

    images = [Image.fromarray(f) for f in frames]
    duration = max(1, int(round(1000.0 / max(1, fps))))  # ms per frame
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,  # loop forever
        disposal=2,  # restore background between frames (no ghosting)
    )


def _save_movie(frames, path, fps) -> None:
    try:
        import imageio
    except ImportError as exc:  # imageio(-ffmpeg) not installed
        raise RuntimeError(
            "Video export needs the 'imageio' and 'imageio-ffmpeg' packages "
            "(pip install imageio imageio-ffmpeg). GIF export needs no extras."
        ) from exc
    writer = imageio.get_writer(path, fps=fps)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


def _save_frame_sequence(frames, path) -> List[str]:
    from PIL import Image

    directory = os.path.dirname(path) or "."
    stem, ext = os.path.splitext(os.path.basename(path))
    written = []
    for i, frame in enumerate(frames):
        out = os.path.join(directory, f"{stem}_{i:03d}{ext}")
        Image.fromarray(frame).save(out)
        written.append(out)
    return written


__all__ = [
    "RASTER_IMAGE_EXTS",
    "VECTOR_IMAGE_EXTS",
    "GIF_EXTS",
    "MOVIE_EXTS",
    "save_view_image",
    "render_animation_frames",
    "save_animation",
]
