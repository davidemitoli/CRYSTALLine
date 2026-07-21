"""QApplication bootstrap."""

from __future__ import annotations

import sys
from typing import Optional

from crystalline.core.structure import Structure


def run(structure: Optional[Structure] = None) -> int:
    """Launch the CRYSTALLine GUI. Returns the Qt exit code."""
    # Imported here so `import crystalline.app` doesn't require a display.
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication
    from crystalline.resources import logo_path
    from crystalline.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CRYSTALLine")
    app.setWindowIcon(QIcon(logo_path()))  # dock / taskbar icon
    window = MainWindow(structure)
    window.show()
    return app.exec()


__all__ = ["run"]
