from __future__ import annotations

from typing import Any, Dict, List

from .base import DexGraspPlugin


class PluginManager:
    def __init__(self) -> None:
        self._plugins: List[DexGraspPlugin] = []

    def register(self, plugin: DexGraspPlugin, task: Any) -> None:
        plugin.on_register(task)
        self._plugins.append(plugin)

    def on_step(self, task: Any) -> None:
        for plugin in self._plugins:
            plugin.on_step(task)

    def finalize(self, task: Any) -> None:
        for plugin in self._plugins:
            plugin.on_finalize(task)

    def notify(self, event_name: str, task: Any, **payload: Dict[str, Any]) -> None:
        for plugin in self._plugins:
            plugin.on_event(event_name, task, payload)








