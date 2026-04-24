from __future__ import annotations

import numpy as np


MOUNTAINCAR_LOW = np.array([-1.2, -0.07], dtype=np.float32)
MOUNTAINCAR_HIGH = np.array([0.6, 0.07], dtype=np.float32)

DINO_LOW = np.array(
    [
        0.0, -20.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)
DINO_HIGH = np.array(
    [
        150.0, 20.0, 1.0, 1.0, 100.0,
        1.0, 600.0, 600.0, 150.0, 150.0,
        1.0, 600.0, 600.0, 150.0, 150.0,
    ],
    dtype=np.float32,
)


def one_hot(index: int, size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    vector[int(index)] = 1.0
    return vector


def normalize_vector(vector: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    clipped = np.clip(vector.astype(np.float32), low, high)
    return (clipped - low) / (high - low + 1e-8)


def get_state_dim(env_name: str, env=None) -> int:
    if env_name == "taxi":
        if env is None:
            raise ValueError("Taxi 环境需要通过 env 获取 one-hot 维度。")
        return int(env.observation_space.n)
    if env_name == "mountaincar":
        return 2
    if env_name == "dino":
        if env is None:
            return int(DINO_LOW.size)
        return int(np.prod(env.observation_space.shape))
    raise ValueError(f"不支持的环境名称: {env_name}")


def process_state(env_name: str, state, env=None) -> np.ndarray:
    # 三种环境最终都会被处理成一维 float32 向量，便于统一输入 MLP。
    if env_name == "taxi":
        if env is None:
            raise ValueError("Taxi 环境需要通过 env 进行 one-hot 编码。")
        return one_hot(int(state), int(env.observation_space.n))

    if env_name == "mountaincar":
        state_array = np.asarray(state, dtype=np.float32)
        return normalize_vector(state_array, MOUNTAINCAR_LOW, MOUNTAINCAR_HIGH)

    if env_name == "dino":
        state_array = np.asarray(state, dtype=np.float32)
        if env is None:
            return normalize_vector(state_array, DINO_LOW, DINO_HIGH)
        return normalize_vector(state_array, env.observation_space.low, env.observation_space.high)

    raise ValueError(f"不支持的环境名称: {env_name}")
