"""Main application window: viewport in the centre, dockable panels around it.

Wiring lives here and nowhere else — panels and the viewport expose signals,
and ``MainWindow`` connects them. Adding a new property (DOS, bands, elastic)
is: build a panel, dock it, connect its signals here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QToolBar,
    QToolButton,
    QWidget,
)

from crystalline.core.cells import (
    CellView,
    as_view,
    complete_boundary,
    expand_modes_to_conventional,
    tile_supercell,
)
from crystalline.core.phonons import PhononModes
from crystalline.core.structure import Structure
from crystalline.core.undo import UndoHistory
from crystalline.viz.phonon_animator import PhononAnimator
from crystalline.ui.viewport import Viewport
from crystalline.ui.panels.structure_panel import StructurePanel
from crystalline.ui.panels.phonon_panel import PhononPanel
from crystalline.ui.panels.info_panel import InfoPanel
from crystalline.ui.panels.display_settings import DisplayPanel
from crystalline.ui.panels.plot_view import PlotPanel


class MainWindow(QMainWindow):
    def __init__(self, structure: Optional[Structure] = None) -> None:
        super().__init__()
        self.setWindowTitle("CRYSTALLine — CRYSTAL structure & phonon viewer")
        from crystalline.resources import logo_path

        self.setWindowIcon(QIcon(logo_path()))
        self.resize(1200, 800)

        # The structure as loaded (CRYSTAL's primitive cell) is the pristine
        # source; ``self.structure`` is the cell view derived from it and shown.
        # By default we show the crystallographic (conventional) cell.
        self._source = structure if structure is not None else Structure.empty()
        self._cell_view = CellView.CRYSTALLOGRAPHIC
        self._supercell = (1, 1, 1)
        self._show_boundary = True  # show partially-belonging molecules by default
        self._editing = False
        self._modes: Optional[PhononModes] = None
        # Undo history for structure edits (snapshots of the shown cell). A view
        # change (supercell/boundary/lattice/load) re-derives from source and
        # resets it — those aren't part of the per-edit undo timeline.
        self._history = UndoHistory()
        self._suppress_undo = False
        self._output_path: Optional[str] = None  # last-loaded CRYSTAL .out, for plots
        self._axis_actions: list = []  # a/b/c view-alignment actions (menu + toolbar)
        self.structure, _, self._unit_cell, self._bond_structure = self._compose_view(
            self._cell_view, self._supercell, None
        )

        # right dock: phonon modes (built before the viewport listener, which
        # references it). Animator drives the renderer.
        self.viewport = Viewport(self)
        self.animator = PhononAnimator(self.viewport.renderer)
        self.phonon_panel = PhononPanel(self.animator, self)

        # centre: 3D viewport
        self.setCentralWidget(self.viewport)
        self.viewport.show_structure(
            self.structure, reference_cell=self._unit_cell, bond_structure=self._bond_structure
        )
        self.structure.add_listener(self._on_structure_changed)

        # The Structure panel is kept as the selection/edit model (it owns the
        # shared selection and backs the Edit-menu tools) but is no longer shown
        # as a dock — 3D picking/drag and the Edit menu drive editing instead.
        self.structure_panel = StructurePanel(self.structure, self)
        self.structure_panel.hide()
        self._dock("Phonons", self.phonon_panel, Qt.RightDockWidgetArea)

        # left dock: crystallographic info of the loaded system
        self.info_panel = InfoPanel(self)
        info_dock = self._dock("Info", self.info_panel, Qt.LeftDockWidgetArea)
        self.info_panel.show_structure(self._source)

        # left dock (tabbed behind Info): live display settings.
        self.display_panel = DisplayPanel(
            self.viewport.renderer.settings, self._apply_render_settings, self
        )
        self._display_dock = self._dock("Display", self.display_panel, Qt.LeftDockWidgetArea)
        self.tabifyDockWidget(info_dock, self._display_dock)
        info_dock.raise_()  # show Info on top by default
        self.display_panel.set_elements(self.structure.numbers)  # initial element swatches

        # bottom dock: property plots (IR/Raman/bands/DOS…), one tab each.
        # Hidden until the first plot is built so it doesn't take up space.
        self.plot_panel = PlotPanel(self)
        self._plot_dock = self._dock("Plots", self.plot_panel, Qt.BottomDockWidgetArea)
        self._plot_dock.hide()

        # Bottom-right status indicators. Order matters: permanent widgets stack
        # left-to-right in call order, so counts sit left of the editing badge.
        self._count_status = QLabel()
        self._count_status.setStyleSheet("color: palette(mid); padding: 0 8px;")
        self.statusBar().addPermanentWidget(self._count_status)

        self._cell_status = QLabel()
        self._cell_status.setStyleSheet("color: palette(mid); padding: 0 8px;")
        self.statusBar().addPermanentWidget(self._cell_status)

        # Shown only while editing is on.
        self._editing_status = QLabel("● Editing mode")
        self._editing_status.setStyleSheet("color: #d9822b; font-weight: bold; padding: 0 8px;")
        self.statusBar().addPermanentWidget(self._editing_status)
        self._editing_status.hide()

        self._connect_signals()
        self._build_menu()
        self._reset_undo()
        self._update_export_actions()
        self._update_status()

    # ── wiring ──────────────────────────────────────────────────────────
    def _on_structure_changed(self, s: Structure) -> None:
        """Model edited: record undo, redraw, and reconcile a running animation."""
        self._capture_undo(s)
        self._update_status()  # atom count may have changed (add/remove)
        self.viewport.renderer.refresh()
        # Editing the geometry: stop any animation and re-anchor it to the edited
        # geometry (drop the modes if the atom count changed). Passing the new
        # positions avoids resetting atoms back to the stale equilibrium, which
        # would undo the edit that was just made.
        self.phonon_panel.invalidate_on_edit(s.positions)

    # ── undo ────────────────────────────────────────────────────────────
    def _capture_undo(self, s: Structure) -> None:
        """Push the pre-edit snapshot so this change can be undone.

        Each edit fires exactly one change notification; the baseline held from
        the previous notification is the state *before* this edit, so it's what
        an undo restores. Skipped while an undo is itself being applied.
        """
        if self._suppress_undo:
            return
        self._history.record(s.to_ase())
        self._update_undo_action()

    def _reset_undo(self) -> None:
        """Clear the undo history and re-baseline to the current structure.

        Used when the shown structure is replaced wholesale (file load, cell
        view, supercell, boundary, lattice parameters): edits don't carry across
        those, so neither does their undo timeline.
        """
        self._history.reset(self.structure.to_ase() if self.structure is not None else None)
        self._update_undo_action()

    def _undo(self) -> None:
        """Revert the most recent structure edit."""
        self._apply_history(self._history.undo())

    def _redo(self) -> None:
        """Re-apply the most recently undone structure edit."""
        self._apply_history(self._history.redo())

    def _apply_history(self, atoms) -> None:
        """Restore a snapshot returned by undo()/redo() (shared plumbing)."""
        if atoms is None:
            return
        self._suppress_undo = True
        try:
            self.structure.restore(atoms)  # listeners redraw; capture suppressed
        finally:
            self._suppress_undo = False
        self.structure_panel.clear_selection()  # indices may no longer be valid
        self._update_undo_action()

    def _update_undo_action(self) -> None:
        undo = getattr(self, "_undo_action", None)
        if undo is not None:
            undo.setEnabled(self._history.can_undo())
        redo = getattr(self, "_redo_action", None)
        if redo is not None:
            redo.setEnabled(self._history.can_redo())

    def _history_icon(self, theme_name: str, standard_pixmap: str):
        """An undo/redo icon: the desktop theme's if present, else a Qt fallback."""
        from PySide6.QtWidgets import QStyle

        icon = QIcon.fromTheme(theme_name)
        if icon.isNull():
            icon = self.style().standardIcon(getattr(QStyle.StandardPixmap, standard_pixmap))
        return icon

    def _connect_signals(self) -> None:
        # viewport pick -> update selection (additive with Ctrl/Shift)
        self.viewport.atom_picked.connect(self.structure_panel.select_atom)
        # dragging an atom in 3D -> update selection (keep a group drag intact)
        self.viewport.atom_moved.connect(self._on_atom_moved)
        # clicking empty space in the 3D view -> clear the panel selection
        self.viewport.selection_cleared.connect(self.structure_panel.clear_selection)
        # starting to drag an atom -> stop any running phonon animation
        self.viewport.interaction_started.connect(self.phonon_panel.stop)
        # the panel owns the selection -> highlight it + refresh the Edit menu
        self.structure_panel.selection_changed.connect(self._on_selection_changed)
        # a phonon mode was (de)selected -> refresh the animation-export action
        self.phonon_panel.mode_selected.connect(lambda _row: self._update_export_actions())

    def _dock(self, title: str, widget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("Open…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        import_action = QAction("Import atoms into structure…", self)
        import_action.triggered.connect(self._import_atoms)
        file_menu.addAction(import_action)

        file_menu.addSeparator()
        save_gui = QAction("Save structure as .gui…", self)
        save_gui.triggered.connect(self._save_gui)
        file_menu.addAction(save_gui)

        save_cif = QAction("Save structure as .cif…", self)
        save_cif.triggered.connect(self._save_cif)
        file_menu.addAction(save_cif)

        file_menu.addSeparator()
        export_image = QAction("Export image…", self)
        export_image.triggered.connect(self._export_image)
        file_menu.addAction(export_image)

        self._export_anim_action = QAction("Export phonon animation…", self)
        self._export_anim_action.triggered.connect(self._export_animation)
        file_menu.addAction(self._export_anim_action)

        self._build_cell_menu()
        self._build_edit_menu()
        self._build_view_menu()
        self._build_plot_menu()
        self._build_help_menu()
        self._build_toolbar()

    def _build_help_menu(self) -> None:
        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("About CRYSTALLine", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_about(self) -> None:
        """A small About dialog showing the logo and version."""
        from PySide6.QtCore import Qt as _Qt
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

        from crystalline.resources import logo_path

        dialog = QDialog(self)
        dialog.setWindowTitle("About CRYSTALLine")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 20, 24, 16)
        layout.setSpacing(12)

        logo = QLabel()
        logo.setPixmap(QPixmap(logo_path()).scaledToWidth(320, _Qt.SmoothTransformation))
        logo.setAlignment(_Qt.AlignCenter)
        layout.addWidget(logo)

        caption = QLabel(
            "<div style='text-align:center'>"
            "<b>CRYSTALLine</b><br>"
            "A desktop viewer &amp; editor for CRYSTAL structures and phonons.<br>"
            "<span style='color:gray'>Built on CRYSTALClear.<br>"
            "Developed with the assistance of Claude (Anthropic).</span></div>"
        )
        caption.setTextFormat(_Qt.RichText)
        caption.setAlignment(_Qt.AlignCenter)
        layout.addWidget(caption)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _build_toolbar(self) -> None:
        """Toolbars for the most-used actions: Undo, and a/b/c view alignment."""
        edit_toolbar = QToolBar("Edit", self)
        edit_toolbar.addAction(self._undo_action)
        edit_toolbar.addAction(self._redo_action)
        self.addToolBar(edit_toolbar)

        view_toolbar = QToolBar("View", self)
        caption = QLabel("View along")
        caption.setContentsMargins(8, 0, 6, 0)
        view_toolbar.addWidget(caption)
        # Colour the a/b/c chips to match the lattice gizmo (a=red, b=green,
        # c=blue), so the button and the on-screen axis arrow read as the same.
        self._axis_buttons: list = []
        for label, axis, color in (("a", 0, "#d62728"), ("b", 1, "#2ca02c"), ("c", 2, "#1f77b4")):
            button = QToolButton(self)
            button.setText(label)
            button.setToolTip(f"Look down the {label} axis")
            button.setStyleSheet(_axis_chip_style(color))
            button.clicked.connect(lambda _checked=False, a=axis: self.viewport.align_view_along(a))
            view_toolbar.addWidget(button)
            self._axis_buttons.append(button)
        view_toolbar.addWidget(self._toolbar_spacer(6))
        self.addToolBar(view_toolbar)
        self._update_view_actions()

    @staticmethod
    def _toolbar_spacer(width: int) -> "QWidget":
        spacer = QWidget()
        spacer.setFixedWidth(width)
        return spacer

    def _update_view_actions(self) -> None:
        """Enable a/b/c view alignment only when there's a cell to align to."""
        enabled = self.viewport.can_align_axes()
        for widget in (*self._axis_actions, *getattr(self, "_axis_buttons", [])):
            widget.setEnabled(enabled)

    def _build_plot_menu(self) -> None:
        """A 'Plot' menu routing CRYSTAL results through CRYSTALClear.plot.

        Output-file plots (IR/Raman/elastic/EOS) read the loaded ``.out``
        directly; data-file plots (bands/DOS/XRD) open a file dialog. Related
        entries (the elastic surfaces) go into a submenu. The whole menu is
        disabled if CRYSTALClear is missing.
        """
        from crystalline.crystalio import available_plots, crystalclear_available

        plot_menu = self.menuBar().addMenu("&Plot")
        self._plot_kinds = available_plots()
        self._plot_actions: dict = {}
        submenus: dict = {}
        for kind in self._plot_kinds:
            target = plot_menu
            if kind.group:
                target = submenus.get(kind.group)
                if target is None:
                    target = plot_menu.addMenu(kind.group)
                    submenus[kind.group] = target
            action = QAction(kind.label, self)
            action.triggered.connect(lambda _checked=False, k=kind: self._open_plot(k))
            target.addAction(action)
            self._plot_actions[kind.key] = action
        if not crystalclear_available():
            plot_menu.setEnabled(False)
            plot_menu.setTitle("&Plot (CRYSTALClear not installed)")
        self._update_plot_actions()

    def _update_plot_actions(self) -> None:
        """Enable each plot action according to the loaded file.

        Output plots (IR/Raman/elastic/EOS) are enabled only when the loaded
        CRYSTAL output actually contains their data; data plots (bands/DOS/XRD)
        stay enabled — they read a separate file the user chooses.
        """
        from crystalline.crystalio import output_availability

        available = output_availability(self._output_path)
        for kind in self._plot_kinds:
            enabled = True if kind.source == "data" else (kind.key in available)
            self._plot_actions[kind.key].setEnabled(enabled)

    def _open_plot(self, kind) -> None:
        """Build ``kind``'s figure and add it as a tab in the Plots dock.

        Output-file plots reuse the loaded CRYSTAL output (prompting only if none
        is loaded); data-file plots always prompt for their PROPERTIES file.
        """
        if kind.source == "output" and self._output_path:
            path = self._output_path
        else:
            path, _ = QFileDialog.getOpenFileName(self, kind.caption, "", kind.file_filter)
            if not path:
                return
        try:
            figure = kind.build(path)
        except Exception as exc:  # noqa: BLE001 - surface any read/plot error to the user
            QMessageBox.critical(self, "Plot failed", f"Could not create the plot:\n{exc}")
            return
        self.plot_panel.add_figure(figure, kind.label.rstrip("… "))
        # Reveal (and raise) the dock so the freshly-built plot is visible,
        # giving it a generous default height so the plot isn't cramped.
        first_reveal = self._plot_dock.isHidden()
        self._plot_dock.show()
        self._plot_dock.raise_()
        if first_reveal:
            self.resizeDocks([self._plot_dock], [max(360, self.height() // 3)], Qt.Vertical)

    def _build_view_menu(self) -> None:
        """A 'View' menu: show the display panel and align the view to an axis."""
        view_menu = self.menuBar().addMenu("&View")
        display_action = QAction("Display settings", self)
        display_action.triggered.connect(self._show_display_panel)
        view_menu.addAction(display_action)

        view_menu.addSeparator()
        for label, axis in (("Along a axis", 0), ("Along b axis", 1), ("Along c axis", 2)):
            action = QAction(label, self)
            action.triggered.connect(lambda _checked=False, a=axis: self.viewport.align_view_along(a))
            view_menu.addAction(action)
            self._axis_actions.append(action)

    def _show_display_panel(self) -> None:
        """Reveal and raise the (dockable) display-settings panel."""
        self._display_dock.show()
        self._display_dock.raise_()

    def _apply_render_settings(self, settings) -> None:
        self.viewport.renderer.set_settings(settings)

    def _build_edit_menu(self) -> None:
        """An 'Edit' menu: turn editing on, select atoms, and run edit tools."""
        edit_menu = self.menuBar().addMenu("&Edit")

        self._undo_action = QAction(self._history_icon("edit-undo", "SP_ArrowBack"), "Undo", self)
        self._undo_action.setShortcut("Ctrl+Z")
        self._undo_action.setToolTip("Undo (Ctrl+Z)")
        self._undo_action.triggered.connect(self._undo)
        self._undo_action.setEnabled(False)
        edit_menu.addAction(self._undo_action)

        self._redo_action = QAction(self._history_icon("edit-redo", "SP_ArrowForward"), "Redo", self)
        self._redo_action.setShortcuts(["Ctrl+Shift+Z", "Ctrl+Y"])
        self._redo_action.setToolTip("Redo (Ctrl+Shift+Z)")
        self._redo_action.triggered.connect(self._redo)
        self._redo_action.setEnabled(False)
        edit_menu.addAction(self._redo_action)
        edit_menu.addSeparator()

        self._edit_mode_action = QAction("Editing mode", self, checkable=True)
        self._edit_mode_action.setShortcut("Ctrl+E")
        self._edit_mode_action.toggled.connect(self._set_editing)
        edit_menu.addAction(self._edit_mode_action)

        edit_menu.addSeparator()
        for text, slot, shortcut in (
            ("Select all", self._select_all, "Ctrl+A"),
            ("Clear selection", self._clear_selection, None),
            ("Invert selection", self._invert_selection, None),
        ):
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(shortcut)
            action.triggered.connect(slot)
            edit_menu.addAction(action)

        edit_menu.addSeparator()
        # These act on the current selection and only while editing is on.
        self._edit_tool_actions = []
        for text, slot, shortcut in (
            ("Delete selected", self._delete_selected, "Del"),
            ("Duplicate selected", self._duplicate_selected, "Ctrl+D"),
            ("Translate selected…", self._translate_selected, None),
            ("Set element of selected…", self._set_element_selected, None),
        ):
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(shortcut)
            action.triggered.connect(slot)
            edit_menu.addAction(action)
            self._edit_tool_actions.append(action)

        edit_menu.addSeparator()
        restore_action = QAction("Restore geometry", self)
        restore_action.setShortcut("Ctrl+R")
        restore_action.triggered.connect(self._restore_geometry)
        edit_menu.addAction(restore_action)
        self._update_edit_actions()

    def _restore_geometry(self) -> None:
        """Discard in-app edits and rebuild the originally-loaded geometry."""
        self.structure_panel.clear_selection()
        self._apply_cell_view()  # re-derives from the pristine source

    # ── editing: mode, selection, tools ─────────────────────────────────
    def _set_editing(self, enabled: bool) -> None:
        self._editing = bool(enabled)
        self.viewport.set_editing_enabled(self._editing)
        self.structure_panel.set_editing_enabled(self._editing)
        self._editing_status.setVisible(self._editing)  # bottom-right indicator
        self._update_edit_actions()

    def _on_atom_moved(self, index: int) -> None:
        """After a drag: keep a multi-atom (group) selection so it can be dragged
        again; for a lone atom, make it the selection so the editor tracks it."""
        if index not in self.structure_panel.selected_indices():
            self.structure_panel.select_atom(index, False)

    def _on_selection_changed(self, indices) -> None:
        self.viewport.set_selection(indices)
        self._update_edit_actions()
        self._update_status()

    def _update_status(self) -> None:
        """Refresh the bottom-right indicators: atom count, selection, cell view."""
        count = getattr(self, "_count_status", None)
        if count is None:
            return
        n = len(self.structure)
        text = f"{n} atom{'s' if n != 1 else ''}"
        selected = len(self.structure_panel.selected_indices())
        if selected:
            text += f"  ·  {selected} selected"
        count.setText(text)

        na, nb, nc = self._supercell
        cell = "" if self._supercell == (1, 1, 1) else f"supercell {na}×{nb}×{nc}"
        self._cell_status.setText(cell)

    def _update_edit_actions(self) -> None:
        """Edit tools need editing on and at least one atom selected."""
        active = self._editing and bool(self.structure_panel.selected_indices())
        for action in self._edit_tool_actions:
            action.setEnabled(active)

    def _select_all(self) -> None:
        self.structure_panel.set_selection(range(len(self.structure)))

    def _clear_selection(self) -> None:
        self.structure_panel.clear_selection()

    def _invert_selection(self) -> None:
        current = set(self.structure_panel.selected_indices())
        self.structure_panel.set_selection(set(range(len(self.structure))) - current)

    def _delete_selected(self) -> None:
        indices = self.structure_panel.selected_indices()
        if not self._editing or not indices:
            return
        self.structure_panel.clear_selection()  # deletion shifts indices; drop them first
        self.structure.remove_atoms(indices)

    def _duplicate_selected(self) -> None:
        indices = self.structure_panel.selected_indices()
        if not self._editing or not indices:
            return
        # Offset the copies so they don't sit exactly on the originals.
        new = self.structure.duplicate_atoms(indices, offset=(1.5, 0.0, 0.0))
        self.structure_panel.set_selection(new)

    def _translate_selected(self) -> None:
        indices = self.structure_panel.selected_indices()
        if not self._editing or not indices:
            return
        vector = self._ask_vector()
        if vector is not None:
            self.structure.translate_atoms(indices, vector)

    def _set_element_selected(self) -> None:
        indices = self.structure_panel.selected_indices()
        if not self._editing or not indices:
            return
        symbol, ok = QInputDialog.getText(self, "Set element", "Element symbol:")
        if not ok or not symbol.strip():
            return
        try:
            self.structure.set_symbols(indices, symbol.strip())
        except ValueError:
            QMessageBox.warning(self, "Unknown element", f"'{symbol}' is not a known element.")

    def _ask_vector(self):
        """Small dialog returning a cartesian (dx, dy, dz) shift, or None."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Translate selected")
        form = QFormLayout(dialog)
        boxes = []
        for axis in ("x", "y", "z"):
            box = QDoubleSpinBox()
            box.setRange(-1000.0, 1000.0)
            box.setDecimals(3)
            box.setSingleStep(0.1)
            form.addRow(f"Δ{axis} (Å)", box)
            boxes.append(box)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        return [box.value() for box in boxes]

    def _build_cell_menu(self) -> None:
        """A 'Cell' menu: supercell and boundary-completion of the crystallographic cell."""
        cell_menu = self.menuBar().addMenu("&Cell")

        self._lattice_action = QAction("Lattice parameters…", self)
        self._lattice_action.triggered.connect(self._open_lattice_dialog)
        cell_menu.addAction(self._lattice_action)

        self._supercell_action = QAction("Supercell…", self)
        self._supercell_action.triggered.connect(self._open_supercell_dialog)
        cell_menu.addAction(self._supercell_action)

        # Show whole molecules/atoms that only partially belong to the cell
        # (their periodic images poke in) — the "packed" view, on by default.
        # Unchecking restricts the view to the cell's own atoms.
        self._boundary_action = QAction("Complete molecules at cell boundary", self, checkable=True)
        self._boundary_action.setChecked(self._show_boundary)
        self._boundary_action.toggled.connect(self._on_boundary_toggled)
        cell_menu.addAction(self._boundary_action)

    def _on_boundary_toggled(self, enabled: bool) -> None:
        self._show_boundary = bool(enabled)
        self._apply_cell_view()

    def _open_lattice_dialog(self) -> None:
        """Edit the crystal's lattice parameters (a, b, c, α, β, γ).

        Applied to the pristine source and re-derived through the same pipeline
        as the other Cell actions, so the current view (conventional cell,
        supercell, boundary completion) is rebuilt on the new lattice. Atoms keep
        their fractional coordinates (the cell is reshaped around them).
        """
        if not self._source.is_periodic or np.allclose(self._source.cell, 0.0):
            QMessageBox.information(
                self,
                "Lattice parameters",
                "The loaded system is not periodic — there is no cell to edit.",
            )
            return

        a, b, c, alpha, beta, gamma = (float(v) for v in self._source.cellpar)
        dialog = QDialog(self)
        dialog.setWindowTitle("Lattice parameters")
        form = QFormLayout(dialog)
        # (label, value, min, max, decimals) — lengths in Å, angles in degrees
        specs = [
            ("a (Å)", a, 0.1, 1000.0, 4),
            ("b (Å)", b, 0.1, 1000.0, 4),
            ("c (Å)", c, 0.1, 1000.0, 4),
            ("α (°)", alpha, 1.0, 179.0, 3),
            ("β (°)", beta, 1.0, 179.0, 3),
            ("γ (°)", gamma, 1.0, 179.0, 3),
        ]
        boxes = []
        for label, value, lo, hi, decimals in specs:
            box = QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setDecimals(decimals)
            box.setSingleStep(0.1)
            box.setValue(value)
            form.addRow(label, box)
            boxes.append(box)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return
        self._source.set_lattice_parameters(*(box.value() for box in boxes))
        self._apply_cell_view()
        self.info_panel.show_structure(self._source)

    def _open_supercell_dialog(self) -> None:
        """Prompt for the na × nb × nc repetitions and rebuild the view.

        The cell wireframe keeps outlining the original unit cell (not the
        enlarged supercell box) — see ``_compose_view``/``Viewport.show_structure``.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Supercell")
        form = QFormLayout(dialog)
        boxes = []
        for axis, value in zip(("a", "b", "c"), self._supercell):
            box = QSpinBox()
            box.setRange(1, 12)
            box.setValue(value)
            form.addRow(f"Repeat along {axis}", box)
            boxes.append(box)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return
        self._supercell = tuple(box.value() for box in boxes)
        na, nb, nc = self._supercell
        suffix = "" if self._supercell == (1, 1, 1) else f"  ({na}×{nb}×{nc})"
        self._supercell_action.setText(f"Supercell…{suffix}")
        self._apply_cell_view()

    # ── file actions ────────────────────────────────────────────────────
    def _open_file(self) -> None:
        """Single open: loads geometry, and phonon modes too if the file has them."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CRYSTAL file", "", "CRYSTAL files (*.out *.gui *.34);;All files (*)"
        )
        if not path:
            return
        try:
            from crystalline.crystalio import load

            result = load(path)
        except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._source = result.structure
        self._modes = result.modes if result.has_phonons else None
        # Remember the output file so property plots (IR/Raman/elastic/EOS) can
        # read it directly; geometry-only files (.gui/.34) carry no such data.
        self._output_path = None if path.lower().endswith((".gui", ".34")) else path
        self._apply_cell_view()
        self._update_info(path)
        self._update_plot_actions()  # enable only the plots this file supports

    def _import_atoms(self) -> None:
        """Read atoms from an .xyz/.pdb/.cif file and add them to the current structure.

        The atoms are appended at their cartesian coordinates (the source cell is
        ignored). It goes through the normal edit path, so it lands in the undo
        history and the view/panels refresh.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Import atoms", "", "Atom files (*.xyz *.pdb *.cif);;All files (*)"
        )
        if not path:
            return
        try:
            from crystalline.crystalio import read_atoms

            atoms = read_atoms(path)
        except Exception as exc:  # noqa: BLE001 - surface any read/parse error
            QMessageBox.critical(self, "Import failed", f"Could not read atoms:\n{exc}")
            return
        symbols = list(atoms.get_chemical_symbols())
        if not symbols:
            QMessageBox.information(self, "Nothing imported", "No atoms found in that file.")
            return
        try:
            new = self.structure.add_atoms(symbols, atoms.get_positions())  # -> undo + redraw
        except Exception as exc:  # noqa: BLE001 - e.g. an unknown element symbol
            QMessageBox.critical(self, "Import failed", f"Could not add the atoms:\n{exc}")
            return
        self.display_panel.set_elements(self.structure.numbers)  # new elements may appear
        # Turn editing on and select the imported atoms so they can be dragged
        # into place as a whole straight away.
        if not self._editing:
            self._edit_mode_action.setChecked(True)  # toggles _set_editing(True)
        self.structure_panel.set_selection(new)

    def _update_info(self, path: str) -> None:
        """Refresh the crystallographic info panel for the loaded system."""
        props = {}
        try:
            from crystalline.crystalio import output_properties

            props = output_properties(path)
        except Exception:  # noqa: BLE001 - never let output parsing break loading
            props = {}
        self.info_panel.show_structure(self._source, props)

    def _save_gui(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CRYSTAL .gui", "", "CRYSTAL gui (*.gui)"
        )
        if not path:
            return
        try:
            from crystalline.crystalio import save_structure_gui

            save_structure_gui(self.structure, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))

    def _save_cif(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save CIF", "structure.cif", "CIF (*.cif)")
        if not path:
            return
        try:
            from crystalline.crystalio import save_structure_cif

            save_structure_cif(self.structure, path)
        except Exception as exc:  # noqa: BLE001 - surface any write error to the user
            QMessageBox.critical(self, "Save failed", f"Could not save the CIF:\n{exc}")

    # ── export (image / animation) ──────────────────────────────────────
    def _export_image(self) -> None:
        """Save the current 3D view as an image, choosing format/scale/transparency."""
        options = self._ask_image_options()
        if options is None:
            return
        ext, label, scale, transparent = options
        path, _ = QFileDialog.getSaveFileName(
            self, "Export image", f"crystal_view.{ext}", f"{label} (*.{ext})"
        )
        if not path:
            return
        try:
            self.viewport.export_image(path, scale=scale, transparent=transparent)
        except Exception as exc:  # noqa: BLE001 - surface any render/write error
            QMessageBox.critical(self, "Export failed", f"Could not save the image:\n{exc}")

    def _ask_image_options(self):
        """Prompt for ``(ext, filter_label, scale, transparent)``; ``None`` if cancelled.

        Scale supersamples raster output (the 3D analogue of DPI); transparency
        needs an alpha channel, so both are greyed out for the formats that can't
        use them (vector, and opaque rasters like JPEG/BMP).
        """
        # (ext, menu label, is_vector, has_alpha)
        formats = [
            ("png", "PNG image", False, True),
            ("jpg", "JPEG image", False, False),
            ("tif", "TIFF image", False, True),
            ("svg", "SVG vector", True, False),
            ("pdf", "PDF vector", True, False),
            ("eps", "EPS vector", True, False),
        ]
        dialog = QDialog(self)
        dialog.setWindowTitle("Export image")
        form = QFormLayout(dialog)

        fmt_box = QComboBox(dialog)
        for ext, label, is_vector, has_alpha in formats:
            fmt_box.addItem(label, (ext, label, is_vector, has_alpha))
        form.addRow("Format:", fmt_box)

        scale_box = QSpinBox(dialog)
        scale_box.setRange(1, 8)
        scale_box.setValue(2)
        scale_box.setPrefix("×")
        scale_box.setToolTip("Supersampling: ×2 renders at twice the on-screen pixels each way.")
        form.addRow("Resolution:", scale_box)

        transparent = QCheckBox("Transparent background", dialog)
        form.addRow("", transparent)

        def sync_enabled() -> None:
            _ext, _label, is_vector, has_alpha = fmt_box.currentData()
            scale_box.setEnabled(not is_vector)
            transparent.setEnabled(not is_vector and has_alpha)

        fmt_box.currentIndexChanged.connect(sync_enabled)
        sync_enabled()

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return None
        ext, label, is_vector, _has_alpha = fmt_box.currentData()
        scale = 1 if is_vector else scale_box.value()
        want_transparent = transparent.isChecked() and transparent.isEnabled()
        return ext, label, scale, want_transparent

    def _export_animation(self) -> None:
        """Render the selected phonon mode over one cycle and save it.

        GIF needs no extra packages; a PNG filename writes a numbered frame
        sequence; MP4 needs imageio-ffmpeg (a clear message says so if missing).
        Rendered off-screen so the live view is untouched, reusing the current
        appearance and camera.
        """
        selection = self.phonon_panel.current_selection()
        if selection is None:
            QMessageBox.information(
                self, "No mode selected", "Select a vibrational mode to export its animation."
            )
            return
        options = self._ask_animation_options()
        if options is None:
            return
        ext, label, window_size, n_frames, fps = options
        path, _ = QFileDialog.getSaveFileName(
            self, "Export phonon animation", f"phonon_mode.{ext}", f"{label} (*.{ext})"
        )
        if not path:
            return
        equilibrium, mode = selection
        from crystalline.viz.export import render_animation_frames, save_animation

        try:
            frames = render_animation_frames(
                self.structure,
                equilibrium,
                mode,
                self.viewport.renderer.settings,
                amplitude=self.animator.amplitude,
                n_frames=n_frames,
                reference_cell=self._unit_cell,
                bond_structure=self._bond_structure,
                camera=self.viewport.camera_position,
                window_size=window_size,
            )
            written = save_animation(frames, path, fps=fps)
        except Exception as exc:  # noqa: BLE001 - surface encode/write errors clearly
            QMessageBox.critical(self, "Export failed", f"Could not save the animation:\n{exc}")
            return
        if len(written) > 1:
            QMessageBox.information(
                self, "Animation exported", f"Wrote {len(written)} frames next to\n{written[0]}"
            )

    def _ask_animation_options(self):
        """Prompt for ``(ext, filter_label, window_size, n_frames, fps)`` or ``None``.

        Resolution sets the off-screen render size; frames control the smoothness
        of one vibration cycle; FPS the playback speed. A PNG target writes a
        numbered frame sequence rather than a single file.
        """
        from crystalline.viz.export import DEFAULT_FPS, DEFAULT_FRAMES

        formats = [
            ("gif", "Animated GIF"),
            ("mp4", "MP4 video"),
            ("png", "PNG frame sequence"),
        ]
        resolutions = [
            ("640 × 480", (640, 480)),
            ("800 × 600", (800, 600)),
            ("1280 × 960", (1280, 960)),
            ("1920 × 1440", (1920, 1440)),
        ]
        dialog = QDialog(self)
        dialog.setWindowTitle("Export phonon animation")
        form = QFormLayout(dialog)

        fmt_box = QComboBox(dialog)
        for ext, label in formats:
            fmt_box.addItem(label, (ext, label))
        form.addRow("Format:", fmt_box)

        res_box = QComboBox(dialog)
        for label, size in resolutions:
            res_box.addItem(label, size)
        res_box.setCurrentIndex(1)  # 800 × 600
        form.addRow("Resolution:", res_box)

        frames_box = QSpinBox(dialog)
        frames_box.setRange(8, 240)
        frames_box.setValue(DEFAULT_FRAMES)
        frames_box.setToolTip("Frames per vibration cycle — more is smoother but larger.")
        form.addRow("Frames:", frames_box)

        fps_box = QSpinBox(dialog)
        fps_box.setRange(1, 60)
        fps_box.setValue(DEFAULT_FPS)
        fps_box.setSuffix(" fps")
        form.addRow("Frame rate:", fps_box)

        # FPS is meaningless for a still-frame sequence; grey it out for PNG.
        def sync_enabled() -> None:
            fps_box.setEnabled(fmt_box.currentData()[0] != "png")

        fmt_box.currentIndexChanged.connect(sync_enabled)
        sync_enabled()

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return None
        ext, label = fmt_box.currentData()
        return ext, label, res_box.currentData(), frames_box.value(), fps_box.value()

    def _update_export_actions(self) -> None:
        """Enable animation export only when a phonon mode is selected."""
        action = getattr(self, "_export_anim_action", None)
        if action is not None:
            action.setEnabled(self.phonon_panel.has_mode())

    # ── cell view (always the crystallographic cell) ────────────────────
    def _apply_cell_view(self) -> None:
        """Rebuild the shown structure (and phonon modes) from the pristine source.

        Regenerating from the source (rather than transforming the currently
        shown cell) keeps the conversion well-defined and doubles as
        "restore geometry": any in-app edits to the shown structure are dropped.
        """
        try:
            shown, modes, unit_cell, analysis = self._compose_view(
                self._cell_view, self._supercell, self._modes
            )
        except Exception as exc:  # noqa: BLE001 - symmetry analysis can fail
            QMessageBox.warning(
                self, "Cell view unavailable", f"Could not build the crystallographic cell:\n{exc}"
            )
            # Fall back to the loaded cell as-is, which needs no analysis.
            shown, modes, unit_cell, analysis = self._compose_view(
                CellView.PRIMITIVE, self._supercell, self._modes
            )

        self._replace_structure(shown, unit_cell, analysis)
        if modes is not None:
            self.phonon_panel.set_modes(self.structure.positions, modes)
        else:
            self.phonon_panel.clear()
        self._update_export_actions()  # modes may have appeared/disappeared

    def _compose_view(self, view: CellView, supercell, modes):
        """Return ``(structure, modes, unit_cell, analysis)`` for the composed view.

        Pipeline: crystallographic cell → supercell tiling → (optional) boundary
        completion. ``analysis`` is the clean periodic cell *before* boundary
        completion — it drives the bond/polyhedra coordination analysis, since
        the packed cell has duplicate images CrystalNN can't handle. ``unit_cell``
        is the pre-tiling cell, so the viewport keeps outlining the original cell.
        ``modes`` is ``None`` when the file has no phonons.
        """
        if view is CellView.CRYSTALLOGRAPHIC and modes is not None:
            # One expansion produces a consistent structure + modes pair.
            base, base_modes = expand_modes_to_conventional(self._source, modes)
        else:
            base, base_modes = as_view(self._source, view), modes
        unit_cell = base.cell.copy()
        # A clean periodic cell (before boundary completion) drives the
        # chemically-aware bond/polyhedra analysis — the packed cell has
        # duplicate images a near-neighbour algorithm can't handle.
        analysis, analysis_modes = tile_supercell(base, supercell, base_modes)
        if self._show_boundary:
            shown, shown_modes = complete_boundary(analysis, analysis_modes)
        else:
            shown, shown_modes = analysis, analysis_modes
        return shown, shown_modes, unit_cell, analysis

    def _replace_structure(self, new: Structure, unit_cell=None, bond_structure=None) -> None:
        """Swap in a structure to display, rebinding the existing panels.

        We rebind rather than recreate widgets so loading a file (or a view
        change) doesn't spawn duplicate docks (and keeps signals intact).
        ``unit_cell`` outlines the original cell; ``bond_structure`` is the clean
        cell used for the bond/polyhedra coordination analysis.
        """
        self.structure = new
        self.structure.add_listener(self._on_structure_changed)
        self.viewport.show_structure(
            self.structure, reference_cell=unit_cell, bond_structure=bond_structure
        )
        self.structure_panel.set_structure(self.structure)
        self._reset_undo()  # edits (and their undo history) don't cross a re-derive
        self._update_view_actions()  # a/b/c alignment depends on the cell just shown
        if hasattr(self, "display_panel"):
            self.display_panel.set_elements(self.structure.numbers)  # refresh element swatches


def _axis_chip_style(color: str) -> str:
    """Qt stylesheet for a rounded, coloured a/b/c axis button (with states)."""
    hover = QColor(color).lighter(115).name()
    pressed = QColor(color).darker(110).name()
    return f"""
        QToolButton {{
            background-color: {color};
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 4px;
            padding: 4px 11px;
            margin: 2px 1px;
        }}
        QToolButton:hover {{ background-color: {hover}; }}
        QToolButton:pressed {{ background-color: {pressed}; }}
        QToolButton:disabled {{ background-color: #cccccc; color: #f0f0f0; }}
    """


__all__ = ["MainWindow"]
