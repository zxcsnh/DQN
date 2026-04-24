from .env import TrexEnv, register_trex_envs

register_trex_envs()

__all__ = ["TrexEnv", "register_trex_envs"]
