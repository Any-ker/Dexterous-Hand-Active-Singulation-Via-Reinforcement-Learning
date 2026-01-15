from __future__ import annotations

from typing import Any, Dict


class DexGraspPlugin:
    """Base class for task-level plugins."""

    def on_register(self, task: Any) -> None:  # pragma: no cover - interface
        pass

    def on_step(self, task: Any) -> None:  # pragma: no cover - interface
        pass

    def on_finalize(self, task: Any) -> None:  # pragma: no cover - interface
        pass

    def on_event(self, event_name: str, task: Any, payload: Dict[str, Any]) -> None:  # pragma: no cover - interface
        pass








