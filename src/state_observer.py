"""
状态观察模块 - 修复版本
"""
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class StateObserver:
    """状态观察器"""

    def __init__(self, n_agents: int, world_size: int):
        self.n_agents = n_agents
        self.world_size = world_size
        self.detection_radius = 4.0

    def get_observations(self, positions: np.ndarray, velocities: np.ndarray,
                         target_position: np.ndarray, desired_positions: np.ndarray,
                         current_formation: str, obstacle_manager) -> List[np.ndarray]:
        """获取每个智能体的观察"""
        observations = []

        try:
            for i in range(self.n_agents):
                # 自身状态
                own_state = np.concatenate([positions[i], velocities[i]])

                # 目标位置相对向量
                target_relative = target_position - positions[i]

                # 期望队形位置相对向量
                formation_relative = desired_positions[i] - positions[i]

                # 检测附近障碍物
                nearby_obstacles = obstacle_manager.detect_nearby_obstacles(
                    positions[i], self.detection_radius)

                # 障碍物信息（最多3个）
                obstacle_info = self._extract_obstacle_info(nearby_obstacles, positions[i])

                # 其他智能体信息
                teammate_info = self._extract_teammate_info(i, positions, velocities)

                # 当前队形独热编码
                formation_encoding = self._encode_formation(current_formation)

                # 组合所有观察
                observation = np.concatenate([
                    own_state,  # 4 - 自身状态
                    target_relative,  # 2 - 目标相对位置
                    formation_relative,  # 2 - 期望队形位置
                    formation_encoding,  # 5 - 当前队形编码
                    obstacle_info,  # 15 - 3个障碍物信息
                    teammate_info  # 4*(n_agents-1) - 其他智能体信息
                ])

                observations.append(observation)

        except Exception as e:
            logger.error(f"获取观察失败: {e}")
            # 返回零观察
            obs_dim = 4 + 2 + 2 + 5 + 15 + 4 * (self.n_agents - 1)
            observations = [np.zeros(obs_dim) for _ in range(self.n_agents)]

        return observations

    def _extract_obstacle_info(self, nearby_obstacles: List[Dict[str, Any]],
                               agent_pos: np.ndarray) -> np.ndarray:
        """提取障碍物信息"""
        obstacle_info = []
        sorted_obstacles = sorted(nearby_obstacles, key=lambda x: x['distance'])[:3]

        for obs in sorted_obstacles:
            rel_pos = obs['position'] - agent_pos  # 修复：使用传入的agent_pos
            if obs['type'] == 'dynamic':
                obstacle_info.extend([*rel_pos, *obs['velocity'], obs['radius']])
            else:
                obstacle_info.extend([*rel_pos, 0.0, 0.0, obs['radius']])

        # 填充到3个障碍物
        while len(sorted_obstacles) < 3:
            obstacle_info.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            sorted_obstacles.append(None)

        return np.array(obstacle_info)

    def _extract_teammate_info(self, agent_idx: int, positions: np.ndarray,
                               velocities: np.ndarray) -> np.ndarray:
        """提取队友信息"""
        teammate_info = []
        for j in range(self.n_agents):
            if j != agent_idx:  # 修复：使用agent_idx而不是未定义的i
                rel_pos = positions[j] - positions[agent_idx]
                rel_vel = velocities[j]
                teammate_info.extend([*rel_pos, *rel_vel])

        return np.array(teammate_info)

    def _encode_formation(self, formation: str) -> np.ndarray:
        """编码当前队形"""
        formations = ['line', 'column', 'triangle', 'square', 'circle']
        encoding = np.zeros(len(formations))

        try:
            idx = formations.index(formation)
            encoding[idx] = 1.0
        except ValueError:
            # 默认使用triangle
            encoding[2] = 1.0

        return encoding