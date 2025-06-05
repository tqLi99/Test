"""
改进的自适应编队环境
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Tuple, Dict, Any, Optional
import logging
import gc

from obstacle_manager import ObstacleManager
from formation_manager import FormationManager
from state_observer import StateObserver

logger = logging.getLogger(__name__)


class ImprovedAdaptiveFormationEnvironment:
    """改进的自适应编队环境"""

    def __init__(self, config):
        # 从配置中获取参数
        self.n_agents = config.environment.n_agents
        self.world_size = config.environment.world_size
        self.max_steps = config.environment.max_steps
        self.agent_radius = config.environment.agent_radius
        self.obstacle_radius = config.environment.obstacle_radius

        self.current_step = 0
        self.state_dim = 4
        self.action_dim = 2

        # 导航目标
        self.target_position = np.array([self.world_size * 0.8, self.world_size * 0.8])

        # 初始化管理器
        self.obstacle_manager = ObstacleManager(
            self.world_size,
            config.environment.n_static_obstacles,
            config.environment.n_dynamic_obstacles,
            self.obstacle_radius
        )

        self.formation_manager = FormationManager(self.n_agents)
        self.state_observer = StateObserver(self.n_agents, self.world_size)

        # 改进的奖励计算器
        self.reward_calculator = ImprovedRewardCalculator(self)

        # 内存管理
        self._render_cache = {}
        self._max_cache_size = 50

        logger.info(f"环境初始化完成: {self.n_agents}个智能体")

    def reset(self) -> List[np.ndarray]:
        """重置环境"""
        try:
            self.current_step = 0

            # 初始化智能体位置
            start_position = np.array([self.world_size * 0.2, self.world_size * 0.2])
            spread = 2.0

            self.positions = np.zeros((self.n_agents, 2))
            for i in range(self.n_agents):
                self.positions[i] = start_position + (np.random.rand(2) * 2 - 1) * spread
                self.positions[i] = np.clip(self.positions[i], 0, self.world_size)

            # 初始速度
            self.velocities = np.zeros((self.n_agents, 2))

            # 初始化前一步距离
            team_center = np.mean(self.positions, axis=0)
            self.prev_distance_to_target = np.linalg.norm(team_center - self.target_position)

            # 初始化障碍物
            self.obstacle_manager.initialize_obstacles(self.positions, self.target_position)

            # 初始化队形
            self.formation_manager.current_formation = 'triangle'

            # 清理渲染缓存
            self._cleanup_render_cache()

            return self._get_observations()

        except Exception as e:
            logger.error(f"环境重置失败: {e}")
            raise

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, Dict[str, Any]]:
        """执行一步"""
        try:
            self.current_step += 1

            # 更新动态障碍物
            self.obstacle_manager.update_dynamic_obstacles()

            # 调整队形
            if self.current_step % 5 == 0:
                self._adapt_formation()

            # 更新智能体状态
            collision_info = self._update_agents(actions)

            # 计算奖励
            rewards = self.reward_calculator.compute_rewards()

            # 检查终止条件
            reached_target = self._check_target_reached()
            collision_occurred = collision_info['occurred']

            done = reached_target or self.current_step >= self.max_steps or collision_occurred

            # 获取观察
            observations = self._get_observations()

            info = {
                'reached_target': reached_target,
                'current_formation': self.formation_manager.current_formation,
                'step': self.current_step,
                'collision_occurred': collision_occurred,
                'collision_type': collision_info.get('type'),
                'collision_agent': collision_info.get('agent')
            }

            return observations, rewards, done, info

        except Exception as e:
            logger.error(f"环境步骤执行失败: {e}")
            # 返回安全的默认值
            return self.reset(), [0.0] * self.n_agents, True, {'error': str(e)}

    def _update_agents(self, actions: List[np.ndarray]) -> Dict[str, Any]:
        """更新智能体状态"""
        collision_info = {'occurred': False, 'type': None, 'agent': None}

        try:
            for i in range(self.n_agents):
                # 限制动作范围
                action = np.clip(actions[i], -1, 1)

                # 更新速度
                self.velocities[i] += action * 0.1

                # 限制速度大小
                speed = np.linalg.norm(self.velocities[i])
                max_speed = 0.5
                if speed > max_speed:
                    self.velocities[i] = self.velocities[i] / speed * max_speed

                # 更新位置
                self.positions[i] += self.velocities[i]

                # 边界检查
                self.positions[i] = np.clip(self.positions[i], 0, self.world_size)

                # 碰撞检测
                has_collision, collision_type = self._check_collision(i)
                if has_collision:
                    collision_info = {
                        'occurred': True,
                        'type': collision_type,
                        'agent': i
                    }
                    break

        except Exception as e:
            logger.error(f"智能体状态更新失败: {e}")

        return collision_info

    def _adapt_formation(self) -> None:
        """调整队形"""
        try:
            center = np.mean(self.positions, axis=0)
            nearby_obstacles = self.obstacle_manager.detect_nearby_obstacles(center, 6.0)
            self.formation_manager.adapt_formation(center, self.target_position, nearby_obstacles)
        except Exception as e:
            logger.error(f"队形调整失败: {e}")

    def _check_collision(self, agent_idx: int) -> Tuple[bool, Optional[str]]:
        """检查碰撞"""
        try:
            # 与静态障碍物碰撞
            for obstacle in self.obstacle_manager.static_obstacles:
                distance = np.linalg.norm(self.positions[agent_idx] - obstacle['position'])
                if distance < (self.agent_radius + obstacle['radius']):
                    return True, 'static_obstacle'

            # 与动态障碍物碰撞
            for obstacle in self.obstacle_manager.dynamic_obstacles:
                distance = np.linalg.norm(self.positions[agent_idx] - obstacle['position'])
                if distance < (self.agent_radius + obstacle['radius']):
                    return True, 'dynamic_obstacle'

            # 与其他智能体碰撞
            for i in range(self.n_agents):
                if i != agent_idx:
                    distance = np.linalg.norm(self.positions[agent_idx] - self.positions[i])
                    if distance < (2 * self.agent_radius):
                        return True, 'agent_collision'

            return False, None

        except Exception as e:
            logger.error(f"碰撞检测失败: {e}")
            return False, None

    def _check_target_reached(self, threshold: float = 3.0) -> bool:
        """检查是否到达目标"""
        try:
            distances = [np.linalg.norm(self.positions[i] - self.target_position)
                         for i in range(self.n_agents)]
            success_threshold = 0.3 * self.n_agents
            agents_reached = sum(1 for d in distances if d < threshold)
            return agents_reached >= success_threshold
        except Exception as e:
            logger.error(f"目标检查失败: {e}")
            return False

    def _get_observations(self) -> List[np.ndarray]:
        """获取观察"""
        try:
            center = np.mean(self.positions, axis=0)
            desired_positions = self.formation_manager.get_desired_formation_positions(
                center, self.target_position)

            return self.state_observer.get_observations(
                self.positions, self.velocities, self.target_position,
                desired_positions, self.formation_manager.current_formation,
                self.obstacle_manager
            )
        except Exception as e:
            logger.error(f"获取观察失败: {e}")
            # 返回零观察
            obs_dim = 4 + 2 + 2 + 5 + 15 + 4 * (self.n_agents - 1)
            return [np.zeros(obs_dim) for _ in range(self.n_agents)]

    def render(self, mode: str = 'human') -> Optional[plt.Figure]:
        """优化的渲染方法"""
        try:
            # 检查缓存
            cache_key = f"{self.current_step}_{mode}"
            if cache_key in self._render_cache:
                return self._render_cache[cache_key]

            fig = plt.figure(figsize=(10, 10))
            plt.xlim(0, self.world_size)
            plt.ylim(0, self.world_size)

            # 绘制目标
            plt.plot(self.target_position[0], self.target_position[1], 'x', color='red', markersize=15)
            plt.text(self.target_position[0], self.target_position[1] + 1, "Target", fontsize=12)

            # 绘制静态障碍物
            for obstacle in self.obstacle_manager.static_obstacles:
                circle = plt.Circle(obstacle['position'], obstacle['radius'], color='gray', alpha=0.7)
                plt.gca().add_artist(circle)

            # 绘制动态障碍物
            for obstacle in self.obstacle_manager.dynamic_obstacles:
                circle = plt.Circle(obstacle['position'], obstacle['radius'], color='orange', alpha=0.7)
                plt.gca().add_artist(circle)

                # 速度向量
                plt.arrow(obstacle['position'][0], obstacle['position'][1],
                          obstacle['velocity'][0] * 3, obstacle['velocity'][1] * 3,
                          head_width=0.3, head_length=0.4, fc='orange', ec='orange', alpha=0.7)

            # 绘制智能体
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

            # 期望队形位置
            center = np.mean(self.positions, axis=0)
            desired_positions = self.formation_manager.get_desired_formation_positions(
                center, self.target_position)

            for i in range(self.n_agents):
                # 期望位置
                circle = plt.Circle(desired_positions[i], self.agent_radius / 2,
                                    color=colors[i % len(colors)], alpha=0.3)
                plt.gca().add_artist(circle)

                # 实际位置
                circle = plt.Circle(self.positions[i], self.agent_radius,
                                    color=colors[i % len(colors)], alpha=0.7)
                plt.gca().add_artist(circle)

                # 速度向量
                plt.arrow(self.positions[i][0], self.positions[i][1],
                          self.velocities[i][0] * 2, self.velocities[i][1] * 2,
                          head_width=0.2, head_length=0.3,
                          fc=colors[i % len(colors)], ec=colors[i % len(colors)])

                # 编号
                plt.text(self.positions[i][0], self.positions[i][1], f'{i + 1}',
                         color='white', fontsize=8, ha='center', va='center')

            plt.title(f'Step: {self.current_step} | Formation: {self.formation_manager.current_formation}')
            plt.grid(True)

            # 缓存管理
            if len(self._render_cache) > self._max_cache_size:
                self._cleanup_render_cache()

            if mode == 'human':
                plt.savefig(f'formation_step_{self.current_step:03d}.png')
                plt.close(fig)
                return None
            elif mode == 'return':
                self._render_cache[cache_key] = fig
                return fig
            else:
                plt.close(fig)
                return None

        except Exception as e:
            logger.error(f"渲染失败: {e}")
            return None

    def _cleanup_render_cache(self) -> None:
        """清理渲染缓存"""
        try:
            for fig in self._render_cache.values():
                if fig is not None:
                    plt.close(fig)
            self._render_cache.clear()
            gc.collect()
        except Exception as e:
            logger.error(f"清理渲染缓存失败: {e}")

    def close(self) -> None:
        """关闭环境"""
        self._cleanup_render_cache()
        logger.info("环境已关闭")

    def __del__(self):
        self.close()


# 改进的奖励计算器
class ImprovedRewardCalculator:
    """改进的奖励计算器"""

    def __init__(self, env):
        self.env = env
        self.reward_weights = {
            'progress': 6.0,
            'formation': 0.2,
            'collision': -3.0,
            'cooperation': 1.0,
            'efficiency': 0.5,
            'target_reach': 3.0,
            'survival': -0.01,
            'success': 10.0
        }

    def compute_rewards(self) -> List[float]:
        """计算奖励"""
        try:
            rewards = np.zeros(self.env.n_agents)

            # 期望队形位置
            center = np.mean(self.env.positions, axis=0)
            desired_positions = self.env.formation_manager.get_desired_formation_positions(
                center, self.env.target_position)

            # 计算到目标的距离
            distance_to_target = np.linalg.norm(center - self.env.target_position)
            prev_distance = getattr(self.env, 'prev_distance_to_target', distance_to_target)
            self.env.prev_distance_to_target = distance_to_target

            # 进展奖励
            progress_reward = (prev_distance - distance_to_target) * self.reward_weights['progress']

            # 合作奖励
            cooperation_reward = self._compute_cooperation_reward()

            # 效率奖励
            efficiency_reward = self._compute_efficiency_reward()

            for i in range(self.env.n_agents):
                # 基础团队奖励
                rewards[i] += progress_reward + cooperation_reward + efficiency_reward

                # 队形保持奖励
                formation_error = np.linalg.norm(self.env.positions[i] - desired_positions[i])
                formation_reward = -self.reward_weights['formation'] * formation_error
                rewards[i] += formation_reward

                # 碰撞惩罚
                has_collision, _ = self.env._check_collision(i)
                if has_collision:
                    rewards[i] += self.reward_weights['collision']

                # 到达目标奖励
                agent_distance = np.linalg.norm(self.env.positions[i] - self.env.target_position)
                if agent_distance < 2.0:
                    rewards[i] += self.reward_weights['target_reach']

                # 存活奖励
                rewards[i] += self.reward_weights['survival']

                # 成功奖励
                if self.env._check_target_reached():
                    rewards[i] += self.reward_weights['success']

            return rewards.tolist()

        except Exception as e:
            logger.error(f"奖励计算失败: {e}")
            return [0.0] * self.env.n_agents

    def _compute_cooperation_reward(self) -> float:
        """计算合作奖励"""
        try:
            cooperation_score = 0
            n_pairs = 0

            for i in range(self.env.n_agents):
                for j in range(i + 1, self.env.n_agents):
                    dist = np.linalg.norm(self.env.positions[i] - self.env.positions[j])
                    optimal_dist = 2.0

                    if 1.5 <= dist <= 3.0:
                        cooperation_score += 1.0 - abs(dist - optimal_dist) / optimal_dist

                    n_pairs += 1

            return cooperation_score / max(n_pairs, 1) * self.reward_weights['cooperation']

        except Exception as e:
            logger.error(f"合作奖励计算失败: {e}")
            return 0.0

    def _compute_efficiency_reward(self) -> float:
        """计算效率奖励"""
        try:
            team_center = np.mean(self.env.positions, axis=0)
            team_velocity = np.mean(self.env.velocities, axis=0)

            direction_to_target = self.env.target_position - team_center
            norm = np.linalg.norm(direction_to_target)
            if norm > 1e-8:
                direction_to_target = direction_to_target / norm
            else:
                direction_to_target = np.array([0, 1])

            velocity_magnitude = np.linalg.norm(team_velocity)
            if velocity_magnitude > 1e-8:
                velocity_direction = team_velocity / velocity_magnitude
                alignment = np.dot(velocity_direction, direction_to_target)
                return max(0, alignment) * velocity_magnitude * self.reward_weights['efficiency']

            return 0.0

        except Exception as e:
            logger.error(f"效率奖励计算失败: {e}")
            return 0.0