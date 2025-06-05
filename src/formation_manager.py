"""
队形管理模块
"""
import numpy as np
import math
from typing import Dict, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


class FormationManager:
    """队形管理器"""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.current_formation = 'triangle'
        self.formation_orientation = np.array([0, 1])

        # 定义队形模板
        self.formation_templates: Dict[str, Callable] = {
            'line': self._generate_line_formation,
            'column': self._generate_column_formation,
            'triangle': self._generate_triangle_formation,
            'square': self._generate_square_formation,
            'circle': self._generate_circle_formation
        }

    def _generate_line_formation(self) -> np.ndarray:
        """生成一字型队形"""
        positions = np.zeros((self.n_agents, 2))
        for i in range(self.n_agents):
            positions[i] = [i - (self.n_agents - 1) / 2, 0]
        return positions

    def _generate_column_formation(self) -> np.ndarray:
        """生成纵队队形"""
        positions = np.zeros((self.n_agents, 2))
        for i in range(self.n_agents):
            positions[i] = [0, i - (self.n_agents - 1) / 2]
        return positions

    def _generate_triangle_formation(self) -> np.ndarray:
        """生成三角队形"""
        positions = np.zeros((self.n_agents, 2))
        if self.n_agents >= 3:
            positions[0] = [0, 0.5]
            if self.n_agents == 3:
                positions[1] = [-0.5, -0.5]
                positions[2] = [0.5, -0.5]
            else:
                base_agents = self.n_agents - 1
                for i in range(base_agents):
                    angle = math.pi * (3 / 4 + i * (1 / 2) / (base_agents - 1))
                    positions[i + 1] = [math.cos(angle), math.sin(angle)]
        else:
            return self._generate_line_formation()
        return positions

    def _generate_square_formation(self) -> np.ndarray:
        """生成方形队形"""
        positions = np.zeros((self.n_agents, 2))
        if self.n_agents >= 4:
            corners = 4
            for i in range(min(corners, self.n_agents)):
                angle = i * 2 * math.pi / corners
                positions[i] = [math.cos(angle), math.sin(angle)]

            if self.n_agents > corners:
                remaining = self.n_agents - corners
                edges_per_side = remaining // 4 + 1
                idx = corners

                for side in range(4):
                    start_corner = positions[side]
                    end_corner = positions[(side + 1) % corners]

                    for j in range(1, edges_per_side):
                        if idx < self.n_agents:
                            t = j / edges_per_side
                            positions[idx] = (1 - t) * start_corner + t * end_corner
                            idx += 1
        else:
            return self._generate_triangle_formation()
        return positions

    def _generate_circle_formation(self) -> np.ndarray:
        """生成圆形队形"""
        positions = np.zeros((self.n_agents, 2))
        for i in range(self.n_agents):
            angle = i * 2 * math.pi / self.n_agents
            positions[i] = [math.cos(angle), math.sin(angle)]
        return positions

    def adapt_formation(self, center: np.ndarray, target_position: np.ndarray,
                        nearby_obstacles: list) -> None:
        """根据环境动态调整队形"""
        try:
            if not nearby_obstacles:
                self.current_formation = 'triangle'
                return

            # 计算到目标的方向
            direction_to_target = target_position - center
            norm = np.linalg.norm(direction_to_target)
            if norm > 1e-8:
                direction_to_target = direction_to_target / norm
            else:
                direction_to_target = np.array([0, 1])

            # 分析障碍物分布
            obstacle_directions = []
            for obstacle in nearby_obstacles:
                direction = obstacle['position'] - center
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction = direction / norm
                    obstacle_directions.append(direction)

            # 统计障碍物方向
            front_count = 0
            side_count = 0
            for direction in obstacle_directions:
                cos_angle = np.dot(direction, direction_to_target)
                if cos_angle > 0.7:  # 前方障碍物
                    front_count += 1
                elif abs(cos_angle) < 0.7:  # 侧方障碍物
                    side_count += 1

            # 根据障碍物分布选择队形
            if front_count > 0 and side_count == 0:
                self.current_formation = 'line'
                self.formation_orientation = np.array([-direction_to_target[1], direction_to_target[0]])
            elif front_count > 0 and side_count > 0:
                self.current_formation = 'circle'
            elif side_count > 0 and front_count == 0:
                self.current_formation = 'column'
                self.formation_orientation = direction_to_target
            else:
                self.current_formation = 'triangle'
                self.formation_orientation = direction_to_target

        except Exception as e:
            logger.error(f"队形适应失败: {e}")
            self.current_formation = 'triangle'

    def get_desired_formation_positions(self, center: np.ndarray,
                                        target_position: np.ndarray,
                                        scale: float = 1.5) -> np.ndarray:
        """获取期望的队形位置"""
        try:
            # 生成相对位置
            formation_func = self.formation_templates[self.current_formation]
            relative_positions = formation_func() * scale

            # 计算到目标的方向
            direction_to_target = target_position - center
            norm = np.linalg.norm(direction_to_target)
            if norm > 1e-8:
                direction_to_target = direction_to_target / norm
            else:
                direction_to_target = np.array([0, 1])

            # 根据前进方向旋转队形
            forward = getattr(self, 'formation_orientation', direction_to_target)
            default_forward = np.array([0, 1])

            # 计算旋转角度
            cos_angle = np.dot(default_forward, forward)
            sin_angle = forward[0] * default_forward[1] - forward[1] * default_forward[0]

            # 旋转矩阵
            rotation_matrix = np.array([
                [cos_angle, -sin_angle],
                [sin_angle, cos_angle]
            ])

            # 应用旋转
            rotated_positions = np.zeros_like(relative_positions)
            for i in range(len(relative_positions)):
                rotated_positions[i] = rotation_matrix @ relative_positions[i]

            # 计算绝对位置
            desired_positions = center + rotated_positions
            return desired_positions

        except Exception as e:
            logger.error(f"计算队形位置失败: {e}")
            # 返回当前中心周围的默认位置
            return np.array([center + np.random.randn(2) * 0.5 for _ in range(self.n_agents)])