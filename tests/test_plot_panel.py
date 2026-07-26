"""The Plots dock: it floats on the first plot, at a figure-friendly aspect.

``MainWindow`` can't be constructed headless (VTK interactor), so the reveal
logic is exercised unbound against a stub window that provides just the bits it
touches — and the geometry maths, which is pure, is tested directly.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRect, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow  # noqa: E402

from crystalline.ui.main_window import MainWindow, _floating_plot_geometry  # noqa: E402

_SCREEN = QRect(0, 0, 2560, 1440)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_floating_plot_geometry_is_four_by_three_and_on_screen():
    for window in (QRect(100, 80, 1200, 800), QRect(0, 0, 2560, 1400), QRect(50, 50, 700, 500)):
        rect = _floating_plot_geometry(window, _SCREEN)
        assert rect.width() / rect.height() == pytest.approx(4 / 3, abs=0.01)
        assert _SCREEN.contains(rect), "the plot window must open fully on screen"
        assert rect.width() >= 640  # never squeezed below a usable width


def test_floating_plot_geometry_scales_with_the_main_window():
    small = _floating_plot_geometry(QRect(0, 0, 1200, 800), _SCREEN)
    large = _floating_plot_geometry(QRect(0, 0, 2400, 1400), _SCREEN)
    assert large.width() > small.width()


def test_floating_plot_geometry_is_clamped_to_a_small_screen():
    screen = QRect(0, 0, 800, 600)
    rect = _floating_plot_geometry(QRect(0, 0, 1200, 800), screen)
    assert screen.contains(rect)


class _StubWindow(QMainWindow):
    """Only what ``_reveal_plot_dock`` touches, so no VTK is needed."""

    def __init__(self) -> None:
        super().__init__()
        self.resize(1200, 800)
        self._plot_dock = QDockWidget("Plots", self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._plot_dock)
        self._plot_dock.hide()
        self._plot_dock_floated = False


def test_first_plot_floats_the_dock_then_leaves_it_alone(qapp):
    window = _StubWindow()
    assert not window._plot_dock.isFloating()  # starts docked at the bottom

    MainWindow._reveal_plot_dock(window)
    assert window._plot_dock.isFloating()  # first plot pops it out
    assert not window._plot_dock.isHidden()
    ratio = window._plot_dock.width() / max(window._plot_dock.height(), 1)
    assert ratio == pytest.approx(4 / 3, abs=0.3)  # roughly square, not a strip

    # The user drags it back; building more plots must not float it again.
    window._plot_dock.setFloating(False)
    MainWindow._reveal_plot_dock(window)
    assert not window._plot_dock.isFloating()
    assert not window._plot_dock.isHidden()  # still revealed and raised
