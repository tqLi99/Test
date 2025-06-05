"""
内存优化的经验回放缓冲区
"""
import numpy as np
import torch
from collections import deque
from typing import Tuple, Optional, List
import logging
import gc

logger = logging.getLogger(__name__)


class OptimizedPrioritizedReplayBuffer:
    """内存优化的优先经验回放缓冲区"""

    def __init__(self, max_size: int = int(1e6), alpha: float = 0.6,
                 beta: float = 0.4, beta_increment: float = 0.001):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        # 使用更高效的存储结构
        self.buffer = deque(maxlen=max_size)
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.position = 0
        self.size = 0

        # 内存管理
        self._memory_threshold = 0.8  # 80%内存使用率阈值

    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """添加经验到缓冲区"""
        try:
            # 检查内存使用
            if self._check_memory_usage():
                self._cleanup_memory()

            max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

            # 压缩状态以节省内存
            compressed_state = self._compress_state(state)
            compressed_next_state = self._compress_state(next_state)

            experience = (compressed_state, action, reward, compressed_next_state, done)

            if self.size < self.max_size:
                self.buffer.append(experience)
                self.size += 1
            else:
                self.buffer[self.position] = experience

            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.max_size

        except Exception as e:
            logger.error(f"添加经验到缓冲区失败: {e}")

    def sample(self, batch_size: int) -> Optional[Tuple]:
        """采样经验"""
        if self.size < batch_size:
            return None

        try:
            priorities = self.priorities[:self.size]
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()

            indices = np.random.choice(self.size, batch_size, p=probabilities)
            samples = [self.buffer[idx] for idx in indices]

            # 重要性采样权重
            weights = (self.size * probabilities[indices]) ** (-self.beta)
            weights /= weights.max()

            # 增加beta
            self.beta = min(1.0, self.beta + self.beta_increment)

            # 解压缩状态
            state_batch = np.array([self._decompress_state(s[0]) for s in samples])
            action_batch = np.array([s[1] for s in samples])
            reward_batch = np.array([s[2] for s in samples])
            next_state_batch = np.array([self._decompress_state(s[3]) for s in samples])
            done_batch = np.array([s[4] for s in samples])

            return (state_batch, action_batch, reward_batch, next_state_batch, done_batch), indices, weights

        except Exception as e:
            logger.error(f"采样经验失败: {e}")
            return None

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """更新优先级"""
        try:
            for idx, priority in zip(indices, priorities):
                if isinstance(priority, (np.ndarray, torch.Tensor)):
                    priority_val = float(priority.item() if hasattr(priority, 'item') else priority.flatten()[0])
                else:
                    priority_val = float(priority)

                self.priorities[idx] = priority_val + 1e-6

        except Exception as e:
            logger.error(f"更新优先级失败: {e}")

    def _compress_state(self, state: np.ndarray) -> np.ndarray:
        """压缩状态以节省内存"""
        # 使用float16减少内存占用
        return state.astype(np.float16)

    def _decompress_state(self, compressed_state: np.ndarray) -> np.ndarray:
        """解压缩状态"""
        return compressed_state.astype(np.float32)

    def _check_memory_usage(self) -> bool:
        """检查内存使用率"""
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent / 100
            return memory_percent > self._memory_threshold
        except ImportError:
            return False

    def _cleanup_memory(self) -> None:
        """清理内存"""
        gc.collect()
        logger.info("执行内存清理")

    def __len__(self) -> int:
        return self.size


class SimpleReplayBuffer:
    """简单的经验回放缓冲区"""

    def __init__(self, max_size: int = int(1e6)):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Optional[Tuple]:
        """采样经验"""
        if len(self.buffer) < batch_size:
            return None

        import random
        batch = random.sample(list(self.buffer), batch_size)

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*batch))

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch), None, None

    def __len__(self) -> int:
        return len(self.buffer)