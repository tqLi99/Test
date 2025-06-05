"""
并行训练环境
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
from typing import List, Tuple, Any, Dict
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class ParallelEnvironmentWrapper:
    """并行环境包装器"""

    def __init__(self, env_class, env_kwargs: Dict[str, Any], n_envs: int = 4,
                 max_workers: int = None):
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.n_envs = n_envs
        self.max_workers = max_workers or min(n_envs, mp.cpu_count())

        # 创建环境
        self.envs = [env_class(**env_kwargs) for _ in range(n_envs)]

        # 多进程相关
        self.pool = None
        self.use_multiprocessing = True

        logger.info(f"创建了{n_envs}个并行环境，使用{self.max_workers}个工作进程")

    def reset_all(self) -> List[List[np.ndarray]]:
        """重置所有环境"""
        try:
            if self.use_multiprocessing and self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=self.max_workers)

            if self.use_multiprocessing:
                futures = [self.pool.submit(self._reset_env, i) for i in range(self.n_envs)]
                results = [future.result() for future in futures]
            else:
                results = [env.reset() for env in self.envs]

            return results

        except Exception as e:
            logger.error(f"重置环境失败: {e}")
            # 回退到串行模式
            return [env.reset() for env in self.envs]

    def step_all(self, actions_list: List[List[np.ndarray]]) -> List[Tuple]:
        """所有环境执行步骤"""
        try:
            if self.use_multiprocessing and self.pool is not None:
                futures = [self.pool.submit(self._step_env, i, actions_list[i])
                           for i in range(self.n_envs)]
                results = [future.result() for future in futures]
            else:
                results = [self.envs[i].step(actions_list[i]) for i in range(self.n_envs)]

            return results

        except Exception as e:
            logger.error(f"环境步骤执行失败: {e}")
            # 回退到串行模式
            return [self.envs[i].step(actions_list[i]) for i in range(self.n_envs)]

    def _reset_env(self, env_idx: int) -> List[np.ndarray]:
        """重置单个环境"""
        return self.envs[env_idx].reset()

    def _step_env(self, env_idx: int, actions: List[np.ndarray]) -> Tuple:
        """单个环境执行步骤"""
        return self.envs[env_idx].step(actions)

    def close(self) -> None:
        """关闭并行环境"""
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None

        # 清理环境
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()

        logger.info("并行环境已关闭")

    def __del__(self):
        self.close()


def create_parallel_envs(env_class, env_kwargs: Dict[str, Any],
                         n_envs: int = 4) -> ParallelEnvironmentWrapper:
    """创建并行环境"""
    return ParallelEnvironmentWrapper(env_class, env_kwargs, n_envs)