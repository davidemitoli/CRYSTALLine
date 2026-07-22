"""A tiny, snapshot-based undo history — Qt-free so it can be unit-tested.

The GUI records a snapshot of the shown structure after every edit; this class
tracks which prior snapshot each edit can fall back to. It is deliberately
agnostic about *what* a snapshot is (the app uses ``ase.Atoms`` copies): it only
stores and hands them back.

Model
-----
``baseline`` is the state as of the last :meth:`record`. When the next edit calls
:meth:`record` with the new state, the old baseline becomes undoable (pushed on
the stack) and the new state becomes the baseline. :meth:`undo` pops the last
baseline and makes it current again. :meth:`reset` starts a fresh timeline (used
when the structure is replaced wholesale — a file load or cell-view change).
"""

from __future__ import annotations

from typing import Any, List, Optional


class UndoHistory:
    """A bounded stack of edit snapshots.

    Parameters
    ----------
    limit:
        Maximum number of undo steps kept; older ones are dropped.
    """

    def __init__(self, limit: int = 100) -> None:
        if limit < 1:
            raise ValueError("undo limit must be >= 1")
        self._limit = limit
        self._stack: List[Any] = []       # past states (undo)
        self._redo: List[Any] = []        # future states (redo), newest last
        self._baseline: Optional[Any] = None

    def reset(self, snapshot: Optional[Any]) -> None:
        """Drop all history and (re)baseline to ``snapshot``."""
        self._stack = []
        self._redo = []
        self._baseline = snapshot

    def record(self, snapshot: Any) -> None:
        """Note that an edit produced ``snapshot``; the prior state becomes undoable.

        A fresh edit invalidates any redo timeline (you can't redo past a new
        branch), so the redo stack is cleared.
        """
        if self._baseline is not None:
            self._stack.append(self._baseline)
            if len(self._stack) > self._limit:
                self._stack.pop(0)  # forget the oldest step
        self._baseline = snapshot
        self._redo = []

    def can_undo(self) -> bool:
        return bool(self._stack)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> Optional[Any]:
        """Pop and return the snapshot to restore, or ``None`` if nothing to undo.

        The current baseline is pushed onto the redo stack so the undo can be
        redone; the restored snapshot becomes the new baseline (restoring it must
        not be recorded again as a fresh edit).
        """
        if not self._stack:
            return None
        if self._baseline is not None:
            self._redo.append(self._baseline)
        snapshot = self._stack.pop()
        self._baseline = snapshot
        return snapshot

    def redo(self) -> Optional[Any]:
        """Pop and return the snapshot to re-apply, or ``None`` if nothing to redo.

        The current baseline goes back onto the undo stack; the popped snapshot
        becomes the new baseline. Restoring it must not be recorded as an edit.
        """
        if not self._redo:
            return None
        if self._baseline is not None:
            self._stack.append(self._baseline)
        snapshot = self._redo.pop()
        self._baseline = snapshot
        return snapshot

    def __len__(self) -> int:
        """Number of available undo steps."""
        return len(self._stack)


__all__ = ["UndoHistory"]
