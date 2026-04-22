from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class Obstacle:
    x: float
    width: float
    height: float
    passed: bool = False


class DinoEnv(gym.Env):
    metadata = {"render_modes": [None, "human"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 800) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.screen_width = 600.0
        self.screen_height = 200.0
        self.dino_x = 80.0
        self.ground_y = 0.0
        self.jump_velocity = 12.0
        self.gravity = 1.0
        self.base_speed = 8.0
        self.speed_increase = 0.003
        self.min_spawn_gap = 220.0
        self.max_spawn_gap = 320.0
        self.survive_reward = 0.1
        self.clear_reward = 1.0
        self.collision_penalty = -5.0

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -20.0, 0.0, 0.0, 20.0, 10.0, 5.0], dtype=np.float32),
            high=np.array([200.0, 20.0, 1.0, 600.0, 120.0, 80.0, 20.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.np_random = None
        self.dino_y = 0.0
        self.dino_velocity = 0.0
        self.on_ground = True
        self.game_speed = self.base_speed
        self.steps = 0
        self.score = 0.0
        self.obstacles_cleared = 0
        self.obstacles: list[Obstacle] = []

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.dino_y = 0.0
        self.dino_velocity = 0.0
        self.on_ground = True
        self.game_speed = self.base_speed
        self.steps = 0
        self.score = 0.0
        self.obstacles_cleared = 0
        self.obstacles = [self._create_obstacle(self.screen_width + 120.0)]
        return self._get_state(), {"obstacles_cleared": self.obstacles_cleared}

    def step(self, action: int):
        # 动作 1 表示起跳，仅允许在落地时触发，避免连续跳跃破坏任务节奏。
        if action == 1 and self.on_ground:
            self.dino_velocity = self.jump_velocity
            self.on_ground = False

        self.steps += 1
        self.game_speed = min(20.0, self.base_speed + self.speed_increase * self.steps)

        # 简化的竖直动力学：速度更新 + 重力回落。
        self.dino_y += self.dino_velocity
        self.dino_velocity -= self.gravity
        if self.dino_y <= self.ground_y:
            self.dino_y = self.ground_y
            self.dino_velocity = 0.0
            self.on_ground = True

        reward = self.survive_reward
        terminated = False
        truncated = self.steps >= self.max_steps

        for obstacle in self.obstacles:
            obstacle.x -= self.game_speed
            if not obstacle.passed and obstacle.x + obstacle.width < self.dino_x:
                obstacle.passed = True
                self.obstacles_cleared += 1
                reward += self.clear_reward

        self.obstacles = [obs for obs in self.obstacles if obs.x + obs.width > 0.0]
        if not self.obstacles:
            self.obstacles.append(self._create_obstacle(self.screen_width))
        elif self.obstacles[-1].x < self.screen_width - self._sample_gap():
            self.obstacles.append(self._create_obstacle(self.screen_width))

        # 碰撞给予负奖励并终止回合，用来突出关键失败经验。
        if self._check_collision():
            reward = self.collision_penalty
            terminated = True

        self.score += reward
        state = self._get_state()
        info = {
            "obstacles_cleared": self.obstacles_cleared,
            "score": self.score,
        }

        if self.render_mode == "human":
            self.render()

        return state, reward, terminated, truncated, info

    def render(self):
        nearest = self._get_nearest_obstacle()
        obstacle_distance = nearest.x - self.dino_x if nearest else -1.0
        print(
            f"Dino(y={self.dino_y:.1f}, vy={self.dino_velocity:.1f}, on_ground={int(self.on_ground)}) "
            f"Obstacle(distance={obstacle_distance:.1f}) score={self.score:.1f} cleared={self.obstacles_cleared}"
        )

    def close(self):
        return None

    def _get_state(self) -> np.ndarray:
        nearest = self._get_nearest_obstacle()
        if nearest is None:
            distance = self.screen_width
            height = 40.0
            width = 20.0
        else:
            distance = max(0.0, nearest.x - self.dino_x)
            height = nearest.height
            width = nearest.width

        return np.array(
            [
                self.dino_y,
                self.dino_velocity,
                1.0 if self.on_ground else 0.0,
                distance,
                height,
                width,
                self.game_speed,
            ],
            dtype=np.float32,
        )

    def _check_collision(self) -> bool:
        dino_width = 28.0
        dino_height = 38.0
        dino_left = self.dino_x
        dino_right = self.dino_x + dino_width
        dino_bottom = self.dino_y
        dino_top = self.dino_y + dino_height

        for obstacle in self.obstacles:
            obstacle_left = obstacle.x
            obstacle_right = obstacle.x + obstacle.width
            obstacle_bottom = 0.0
            obstacle_top = obstacle.height

            overlap_x = dino_right > obstacle_left and dino_left < obstacle_right
            overlap_y = dino_top > obstacle_bottom and dino_bottom < obstacle_top
            if overlap_x and overlap_y:
                return True
        return False

    def _get_nearest_obstacle(self) -> Optional[Obstacle]:
        valid_obstacles = [obs for obs in self.obstacles if obs.x + obs.width >= self.dino_x]
        if not valid_obstacles:
            return None
        return min(valid_obstacles, key=lambda obs: obs.x)

    def _sample_gap(self) -> float:
        return float(self.np_random.uniform(self.min_spawn_gap, self.max_spawn_gap))

    def _create_obstacle(self, x_position: float) -> Obstacle:
        width = float(self.np_random.uniform(18.0, 40.0))
        height = float(self.np_random.uniform(30.0, 70.0))
        return Obstacle(x=x_position, width=width, height=height)
