from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from isaacgym import gymapi

from ..interfaces import RunMetadata, VideoLoggingConfig, VisualizationConfig
from .base import DexGraspPlugin

matplotlib.use("Agg")


class TrainingVisualizationLogger:
    def __init__(
        self,
        run_name: str,
        task_name: str,
        reward_mode: str,
        curriculum_enabled: bool,
        obstacle_count: int,
        training_strategy: str,
        output_dir: str,
        plot_interval: int = 200,
        max_buffer: int = 5000,
    ):
        self.run_name = run_name
        self.task_name = task_name
        self.reward_mode = reward_mode
        self.curriculum_enabled = curriculum_enabled
        self.obstacle_count = obstacle_count
        self.training_strategy = training_strategy
        self.plot_interval = max(1, plot_interval)
        self.max_buffer = max_buffer
        self.output_dir = output_dir

        self.learning_steps = []
        self.mean_rewards = []
        self.success_rates = []
        self.ablation_steps = []
        self.ablation_success_rates = []
        self.curriculum_points = []

        self.learning_dir = os.path.join(self.output_dir, "learning_curves")
        self.ablation_dir = os.path.join(self.output_dir, "reward_ablation")
        self.curriculum_dir = os.path.join(self.output_dir, "curriculum")
        self.baseline_dir = os.path.join(self.output_dir, "baseline")
        self.summary_dir = os.path.join(self.output_dir, "summaries")

        for path in [
            self.learning_dir,
            self.ablation_dir,
            self.curriculum_dir,
            self.baseline_dir,
            self.summary_dir,
        ]:
            os.makedirs(path, exist_ok=True)

        self.learning_curve_path = os.path.join(self.learning_dir, f"{self.task_name}_{self.run_name}_learning.png")
        self.learning_raw_path = os.path.join(self.learning_dir, f"{self.task_name}_{self.run_name}_learning.npz")
        self.summary_path = os.path.join(self.summary_dir, f"{self.task_name}_{self.run_name}_summary.json")

    def record_learning_step(self, step, mean_reward, success_rate):
        self.learning_steps.append(step)
        self.mean_rewards.append(mean_reward)
        self.success_rates.append(success_rate)
        self._trim_buffer()

        if len(self.learning_steps) % self.plot_interval == 0:
            self.plot_learning_curves()

    def record_reward_ablation_step(self, step, success_rate):
        if self.obstacle_count != 0:
            return
        self.ablation_steps.append(step)
        self.ablation_success_rates.append(success_rate)
        if len(self.ablation_steps) > self.max_buffer:
            self.ablation_steps = self.ablation_steps[-self.max_buffer :]
            self.ablation_success_rates = self.ablation_success_rates[-self.max_buffer :]

    def record_curriculum_checkpoint(self, step, success_rate):
        self.curriculum_points.append((step, success_rate))
        if len(self.curriculum_points) > self.max_buffer:
            self.curriculum_points = self.curriculum_points[-self.max_buffer :]

    def _trim_buffer(self):
        if len(self.learning_steps) <= self.max_buffer:
            return
        self.learning_steps = self.learning_steps[-self.max_buffer :]
        self.mean_rewards = self.mean_rewards[-self.max_buffer :]
        self.success_rates = self.success_rates[-self.max_buffer :]

    def plot_learning_curves(self, force=False):
        if len(self.learning_steps) < 2 and not force:
            return
        if len(self.learning_steps) == 0:
            return

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(self.learning_steps, self.mean_rewards, color="tab:blue", label="Mean Reward")
        ax1.set_xlabel("Training Steps")
        ax1.set_ylabel("Mean Reward", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(self.learning_steps, self.success_rates, color="tab:orange", label="Success Rate")
        ax2.set_ylabel("Success Rate", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_ylim(0.0, 1.05)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="lower right")

        ax1.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(self.learning_curve_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def finalize(self, final_success_rate, extra_stats=None):
        extra_stats = extra_stats or {}
        self.plot_learning_curves(force=True)
        self._persist_learning_data()
        summary_payload = {
            "task": self.task_name,
            "run_name": self.run_name,
            "reward_mode": self.reward_mode,
            "curriculum_enabled": self.curriculum_enabled,
            "obstacle_count": self.obstacle_count,
            "training_strategy": self.training_strategy,
            "final_success_rate": float(final_success_rate) if final_success_rate is not None else None,
            "timestamp": datetime.utcnow().isoformat(),
            "learning_curve_file": self.learning_curve_path,
            "learning_curve_data": self.learning_raw_path,
            "extra_stats": extra_stats,
        }
        with open(self.summary_path, "w") as f:
            import json

            json.dump(summary_payload, f, indent=2)

        if final_success_rate is not None:
            self._save_reward_ablation_summary(final_success_rate)
            self._plot_curriculum_results()
            self._plot_baseline_results()

    def _persist_learning_data(self):
        if len(self.learning_steps) == 0:
            return
        np.savez(
            self.learning_raw_path,
            steps=np.array(self.learning_steps),
            mean_rewards=np.array(self.mean_rewards),
            success_rates=np.array(self.success_rates),
        )

    def _save_reward_ablation_summary(self, final_success_rate):
        import json

        payload = {
            "task": self.task_name,
            "run_name": self.run_name,
            "reward_mode": self.reward_mode,
            "obstacle_count": self.obstacle_count,
            "final_success_rate": float(final_success_rate),
            "timestamp": datetime.utcnow().isoformat(),
        }
        filename = f"{self.task_name}_{self.reward_mode}_{self.run_name}.json"
        with open(os.path.join(self.ablation_dir, filename), "w") as f:
            json.dump(payload, f, indent=2)
        self._plot_reward_ablation()

    def _plot_reward_ablation(self):
        import glob
        import json

        files = glob.glob(os.path.join(self.ablation_dir, f"{self.task_name}_*.json"))
        reward_groups = {}
        for file_path in files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                if data.get("obstacle_count", 0) != 0:
                    continue
                reward_groups.setdefault(data["reward_mode"], []).append(data["final_success_rate"])
            except Exception:
                continue
        if len(reward_groups) < 2:
            return

        modes = sorted(reward_groups.keys())
        means = [np.mean(reward_groups[m]) for m in modes]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(modes, means, color=["#5DA5DA", "#F15854", "#60BD68", "#B276B2"][: len(modes)])
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Reward Ablation: Success Rate (Single Object)")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.ablation_dir, f"{self.task_name}_reward_ablation.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_curriculum_results(self):
        import glob
        import json

        files = glob.glob(os.path.join(self.summary_dir, f"{self.task_name}_*.json"))
        if not files:
            return
        curriculum_data: Dict[bool, Dict[int, list]] = {}
        for file_path in files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue
            key = data.get("curriculum_enabled")
            if key is None:
                continue
            curriculum_data.setdefault(key, {}).setdefault(data.get("obstacle_count", 0), []).append(
                data.get("final_success_rate", 0.0)
            )

        if not curriculum_data:
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = {True: "tab:green", False: "tab:red"}
        labels = {True: "Curriculum", False: "No Curriculum"}

        for enabled, obstacle_map in curriculum_data.items():
            xs = sorted(obstacle_map.keys())
            ys = [np.mean(obstacle_map[x]) for x in xs]
            ax.plot(xs, ys, marker="o", color=colors.get(enabled, "tab:blue"), label=labels.get(enabled, str(enabled)))

        ax.set_xlabel("Obstacle Count")
        ax.set_ylabel("Final Success Rate")
        ax.set_title("Curriculum Learning Results")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.curriculum_dir, f"{self.task_name}_curriculum.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_baseline_results(self):
        import glob
        import json

        files = glob.glob(os.path.join(self.summary_dir, f"{self.task_name}_*.json"))
        if not files:
            return
        obstacle_strategy: Dict[int, Dict[str, list]] = {}
        for file_path in files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue
            obstacle = data.get("obstacle_count")
            strategy = data.get("training_strategy")
            success = data.get("final_success_rate")
            if obstacle is None or strategy is None or success is None:
                continue
            obstacle_strategy.setdefault(obstacle, {}).setdefault(strategy, []).append(success)

        for obstacle, strategy_map in obstacle_strategy.items():
            if len(strategy_map) < 2:
                continue
            strategies = sorted(strategy_map.keys())
            means = [np.mean(strategy_map[s]) for s in strategies]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(strategies, means, color="#4E79A7")
            ax.set_ylabel("Success Rate")
            ax.set_ylim(0.0, 1.05)
            ax.set_title(f"Baseline Comparison @ Obstacle={obstacle}")
            ax.grid(True, linestyle="--", alpha=0.3)
            fig.tight_layout()
            ax.figure.savefig(
                os.path.join(self.baseline_dir, f"{self.task_name}_baseline_obstacle_{obstacle}.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)


class VisualizationPlugin(DexGraspPlugin):
    def __init__(self, config: VisualizationConfig, metadata: RunMetadata):
        self.config = config
        self.metadata = metadata
        self.logger: TrainingVisualizationLogger | None = None

    def on_register(self, task: Any) -> None:
        if not self.config.enable:
            return
        output_dir = self.config.output_dir or getattr(task, "visualization_output_dir", os.getcwd())
        os.makedirs(output_dir, exist_ok=True)
        self.logger = TrainingVisualizationLogger(
            run_name=self.metadata.run_name,
            task_name=self.metadata.task_name,
            reward_mode=self.metadata.reward_mode,
            curriculum_enabled=self.metadata.curriculum_enabled,
            obstacle_count=self.metadata.obstacle_count,
            training_strategy=self.metadata.training_strategy,
            output_dir=output_dir,
            plot_interval=self.config.plot_interval,
            max_buffer=self.config.max_buffer,
        )

    def on_step(self, task: Any) -> None:
        if not self.logger:
            return
        if not hasattr(task, "rew_buf") or task.rew_buf is None:
            return
        mean_reward = float(task.rew_buf.mean().item()) if task.rew_buf.numel() > 0 else 0.0
        success_tensor = task.successes if hasattr(task, "successes") else None
        success_rate = float(success_tensor.float().mean().item()) if success_tensor is not None else 0.0
        self.logger.record_learning_step(task.global_step, mean_reward, success_rate)
        self.logger.record_reward_ablation_step(task.global_step, success_rate)

    def on_event(self, event_name: str, task: Any, payload: Dict[str, Any]) -> None:
        if not self.logger:
            return
        if event_name == "curriculum_checkpoint":
            self.logger.record_curriculum_checkpoint(payload.get("step", 0), payload.get("success_rate", 0.0))

    def on_finalize(self, task: Any) -> None:
        if not self.logger:
            return
        final_success_rate = float(getattr(task, "current_success_rate", 0.0))
        if np.isnan(final_success_rate):
            final_success_rate = 0.0
        extra_stats = {
            "current_success_rate": final_success_rate,
            "cumulative_success_rate": float(getattr(task, "cumulative_successes", 0.0)),
            "avg_success_steps": getattr(task, "avg_succ_steps", 0),
            "singulation_success_rate": getattr(task, "singulation_success_rate", 0.0),
            "videos": getattr(task, "completed_video_manifest", []),
        }
        self.logger.finalize(final_success_rate, extra_stats=extra_stats)


class VideoCaptureManager:
    def __init__(
        self,
        gym_instance,
        sim,
        envs,
        camera_handles,
        cfg: VideoLoggingConfig,
        run_name: str,
        reward_mode: str,
        output_dir: str,
    ):
        self.gym = gym_instance
        self.sim = sim
        self.envs = envs
        self.camera_handles = camera_handles
        self.cfg = cfg
        self.run_name = run_name
        self.reward_mode = reward_mode
        self.output_dir = os.path.join(output_dir, "videos")
        os.makedirs(self.output_dir, exist_ok=True)
        self.scenes = self._build_scenes(cfg.scenes)
        self.frame_buffers = {scene["name"]: [] for scene in self.scenes}
        self.completed_videos = []

    def _build_scenes(self, scenes_cfg):
        scenes = []
        for scene in scenes_cfg:
            scenes.append(
                {
                    "name": scene.name,
                    "env_id": scene.env_id,
                    "start_step": scene.start_step,
                    "end_step": scene.end_step,
                    "fps": scene.fps,
                    "width": scene.width,
                    "height": scene.height,
                    "codec": scene.codec,
                }
            )
        return scenes

    def capture_step(self, step):
        if not self.scenes:
            return
        active_scenes = [scene for scene in self.scenes if scene["start_step"] <= step <= scene["end_step"]]
        if not active_scenes:
            return

        self.gym.render_all_camera_sensors(self.sim)
        for scene in active_scenes:
            env_id = scene["env_id"]
            camera_handle = self.camera_handles.get(env_id)
            if camera_handle is None:
                continue
            image = self.gym.get_camera_image(self.sim, self.envs[env_id], camera_handle, gymapi.IMAGE_COLOR)
            if image is None:
                continue
            np_image = np.array(image, dtype=np.uint8).reshape(scene["height"], scene["width"], 4)
            rgb = np_image[..., :3]
            self.frame_buffers[scene["name"]].append(rgb)
            if step == scene["end_step"]:
                self._write_video(scene)

    def _write_video(self, scene):
        frames = self.frame_buffers.get(scene["name"], [])
        if not frames:
            return
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{scene['name']}_{self.reward_mode}_{self.run_name}_{timestamp}.mp4"
        save_path = os.path.join(self.output_dir, filename)
        try:
            imageio.mimsave(save_path, frames, fps=scene["fps"])
            self.completed_videos.append({"tag": scene["name"], "path": save_path, "env_id": scene["env_id"]})
        except Exception as e:  # pragma: no cover
            print(f"[VideoCaptureManager] Failed to write video {filename}: {e}")
        finally:
            self.frame_buffers[scene["name"]] = []

    def finalize(self):
        for scene in self.scenes:
            if self.frame_buffers.get(scene["name"]):
                self._write_video(scene)
        return self.completed_videos


class VideoCapturePlugin(DexGraspPlugin):
    def __init__(self, config: VideoLoggingConfig, metadata: RunMetadata):
        self.config = config
        self.metadata = metadata
        self.manager: VideoCaptureManager | None = None

    def _ensure_manager(self, task: Any) -> None:
        if self.manager or not self.config.enable:
            return
        if not getattr(task, "camera_handles_map", None):
            return
        output_dir = getattr(task, "visualization_output_dir", os.getcwd())
        self.manager = VideoCaptureManager(
            gym_instance=task.gym,
            sim=task.sim,
            envs=task.envs,
            camera_handles=task.camera_handles_map,
            cfg=self.config,
            run_name=self.metadata.run_name,
            reward_mode=self.metadata.reward_mode,
            output_dir=output_dir,
        )

    def on_step(self, task: Any) -> None:
        self._ensure_manager(task)
        if self.manager:
            self.manager.capture_step(task.global_step)

    def on_finalize(self, task: Any) -> None:
        if self.manager:
            task.completed_video_manifest = self.manager.finalize()


