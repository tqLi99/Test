"""
障碍物管理模块
"""
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ObstacleManager:
    """障碍物管理器"""

    def __init__(self, world_size: int, n_static: int, n_dynamic: int,
                 obstacle_radius: float = 1.0):
        self.world_size = world_size
        self.n_static = n_static
        self.n_dynamic = n_dynamic
        self.obstacle_radius = obstacle_radius

        self.static_obstacles = []
        self.dynamic_obstacles = []

    def initialize_obstacles(self, agent_positions: np.ndarray = None,
                             target_position: np.ndarray = None) -> None:
        """初始化障碍物"""
        try:
            self.static_obstacles = self._initialize_static_obstacles(
                agent_positions, target_position)
            self.dynamic_obstacles = self._initialize_dynamic_obstacles(
                agent_positions, target_position)
            logger.info(f"初始化完成: {len(self.static_obstacles)}个静态障碍物, "
                        f"{len(self.dynamic_obstacles)}个动态障碍物")
        except Exception as e:
            logger.error(f"障碍物初始化失败: {e}")
            raise

    def _initialize_static_obstacles(self, agent_positions: np.ndarray = None,
                                     target_position: np.ndarray = None) -> List[Dict[str, Any]]:
        """初始化静态障碍物"""
        # 固定障碍物配置
        fixed_obstacles = [
            {'position': np.array([7.5, 12.5]), 'radius': 2.0},
            {'position': np.array([5.0, 7.0]), 'radius': 1.2},
            {'position': np.array([12.5, 7.5]), 'radius': 1.5},
        ]

        n_fixed = min(len(fixed_obstacles), self.n_static)
        obstacles = fixed_obstacles[:n_fixed]

        # 随机生成剩余障碍物
        n_random = self.n_static - n_fixed
        if n_random > 0:
            existing_positions = [obs['position'] for obs in obstacles]
            if agent_positions is not None:
                existing_positions.extend(agent_positions)
            if target_position is not None:
                existing_positions.append(target_position)

            obstacles.extend(self._generate_random_static_obstacles(
                n_random, existing_positions))

        return obstacles

    def _generate_random_static_obstacles(self, n_obstacles: int,
                                          existing_positions: List[np.ndarray]) -> List[Dict[str, Any]]:
        """生成随机静态障碍物"""
        obstacles = []
        safe_radius = 3.0

        for _ in range(n_obstacles):
            valid_position = False
            attempts = 0

            while not valid_position and attempts < 100:
                pos = np.random.rand(2) * self.world_size

                # 检查与现有位置的距离
                valid = True
                for existing_pos in existing_positions:
                    if np.linalg.norm(pos - existing_pos) < safe_radius:
                        valid = False
                        break

                if valid:
                    valid_position = True
                    obstacles.append({
                        'position': pos,
                        'radius': self.obstacle_radius + np.random.rand() * 0.5
                    })
                    existing_positions.append(pos)

                attempts += 1

            if not valid_position:
                logger.warning(f"无法找到合适位置放置第{len(obstacles) + 1}个静态障碍物")

        return obstacles

    def _initialize_dynamic_obstacles(self, agent_positions: np.ndarray = None,
                                      target_position: np.ndarray = None) -> List[Dict[str, Any]]:
        """初始化动态障碍物"""
        fixed_obstacles = [
            {
                'position': np.array([8.0, 15.0]),
                'radius': 0.8,
                'velocity': np.array([0.05, -0.05])
            },
            {
                'position': np.array([15.0, 8.0]),
                'radius': 0.8,
                'velocity': np.array([-0.05, 0.05])
            },
        ]

        n_fixed = min(len(fixed_obstacles), self.n_dynamic)
        obstacles = fixed_obstacles[:n_fixed]

        # 随机生成剩余动态障碍物
        n_random = self.n_dynamic - n_fixed
        if n_random > 0:
            existing_positions = [obs['position'] for obs in obstacles]
            existing_positions.extend([obs['position'] for obs in self.static_obstacles])
            if agent_positions is not None:
                existing_positions.extend(agent_positions)
            if target_position is not None:
                existing_positions.append(target_position)

            obstacles.extend(self._generate_random_dynamic_obstacles(
                n_random, existing_positions))

        return obstacles

    def _generate_random_dynamic_obstacles(self, n_obstacles: int,
                                           existing_positions: List[np.ndarray]) -> List[Dict[str, Any]]:
        """生成随机动态障碍物"""
        obstacles = []
        safe_radius = 3.0

        for _ in range(n_obstacles):
            valid_position = False
            attempts = 0

            while not valid_position and attempts < 100:
                pos = np.random.rand(2) * self.world_size

                valid = True
                for existing_pos in existing_positions:
                    if np.linalg.norm(pos - existing_pos) < safe_radius:
                        valid = False
                        break

                if valid:
                    valid_position = True
                    angle = np.random.rand() * 2 * np.pi
                    speed = 0.1 + np.random.rand() * 0.2
                    velocity = np.array([np.cos(angle), np.sin(angle)]) * speed

                    obstacles.append({
                        'position': pos,
                        'radius': self.obstacle_radius,
                        'velocity': velocity
                    })
                    existing_positions.append(pos)

                attempts += 1

        return obstacles

    def update_dynamic_obstacles(self) -> None:
        """更新动态障碍物位置"""
        for obstacle in self.dynamic_obstacles:
            # 更新位置
            obstacle['position'] += obstacle['velocity']

            # 边界检查和反弹
            for i in range(2):
                if obstacle['position'][i] <= 0 or obstacle['position'][i] >= self.world_size:
                    obstacle['velocity'][i] *= -1
                    obstacle['position'][i] = np.clip(obstacle['position'][i], 0, self.world_size)

            # 随机改变方向
            if np.random.rand() < 0.02:
                angle = np.random.rand() * 2 * np.pi
                speed = np.linalg.norm(obstacle['velocity'])
                obstacle['velocity'] = np.array([np.cos(angle), np.sin(angle)]) * speed

    def detect_nearby_obstacles(self, position: np.ndarray,
                                detection_radius: float = 5.0) -> List[Dict[str, Any]]:
        """检测附近的障碍物"""
        nearby_obstacles = []

        # 检测静态障碍物
        for obstacle in self.static_obstacles:
            dist = np.linalg.norm(position - obstacle['position'])
            if dist < detection_radius:
                nearby_obstacles.append({
                    'position': obstacle['position'],
                    'radius': obstacle['radius'],
                    'distance': dist,
                    'type': 'static'
                })

        # 检测动态障碍物
        for obstacle in self.dynamic_obstacles:
            dist = np.linalg.norm(position - obstacle['position'])
            if dist < detection_radius:
                nearby_obstacles.append({
                    'position': obstacle['position'],
                    'radius': obstacle['radius'],
                    'distance': dist,
                    'velocity': obstacle['velocity'],
                    'type': 'dynamic'
                })

        return nearby_obstacles