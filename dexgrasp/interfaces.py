from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class VisualizationConfig:
    enable: bool = True
    output_dir: str = ""
    plot_interval: int = 200
    max_buffer: int = 5000

    @staticmethod
    def from_dict(cfg: dict, default_dir: str) -> "VisualizationConfig":
        cfg = cfg or {}
        return VisualizationConfig(
            enable=cfg.get("enable", True),
            output_dir=cfg.get("output_dir", default_dir),
            plot_interval=cfg.get("plot_interval", 200),
            max_buffer=cfg.get("max_buffer", 5000),
        )


@dataclass
class VideoSceneConfig:
    name: str
    env_id: int
    start_step: int
    end_step: int
    fps: int = 30
    width: int = 640
    height: int = 480
    codec: str = "mp4"

    @staticmethod
    def from_dict(scene: dict, default_width: int, default_height: int) -> "VideoSceneConfig":
        duration = scene.get("duration", 240)
        start = scene.get("start_step", 0)
        end = scene.get("end_step", start + duration)
        return VideoSceneConfig(
            name=scene["name"],
            env_id=scene["env_id"],
            start_step=start,
            end_step=end,
            fps=scene.get("fps", 30),
            width=scene.get("width", default_width),
            height=scene.get("height", default_height),
            codec=scene.get("codec", "mp4"),
        )


@dataclass
class VideoLoggingConfig:
    enable: bool = False
    width: int = 640
    height: int = 480
    default_pos: Sequence[float] = field(default_factory=lambda: (0.8, 0.0, 1.5))
    default_target: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.6))
    scenes: List[VideoSceneConfig] = field(default_factory=list)
    camera_env_ids: Optional[Sequence[int]] = None
    camera_setups: Optional[Sequence[dict]] = None
    output_dir: Optional[str] = None

    @staticmethod
    def from_dict(cfg: dict) -> "VideoLoggingConfig":
        cfg = cfg or {}
        scene_cfgs = [
            VideoSceneConfig.from_dict(scene, cfg.get("width", 640), cfg.get("height", 480))
            for scene in cfg.get("scenes", [])
        ]
        return VideoLoggingConfig(
            enable=cfg.get("enable", False),
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            default_pos=tuple(cfg.get("default_pos", (0.8, 0.0, 1.5))),
            default_target=tuple(cfg.get("default_target", (0.0, 0.0, 0.6))),
            scenes=scene_cfgs,
            camera_env_ids=cfg.get("camera_env_ids"),
            camera_setups=cfg.get("camera_setups"),
            output_dir=cfg.get("output_dir"),
        )


@dataclass
class RunMetadata:
    run_name: str
    task_name: str
    reward_mode: str
    curriculum_enabled: bool
    obstacle_count: int
    training_strategy: str


