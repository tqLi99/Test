import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import math
import random
import os
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 启用梯度异常检测 (可选，用于调试)
# torch.autograd.set_detect_anomaly(True)

# 创建TensorBoard日志目录
def create_tensorboard_writer(comment=''):
    """创建TensorBoard日志写入器"""
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'{current_time}_{comment}')
    return SummaryWriter(log_dir=log_dir)


class AdaptiveFormationEnvironment:
    """
    具有动态和静态障碍物的多智能体编队环境
    队形会根据障碍物分布动态调整
    """

    def __init__(self, n_agents=3, world_size=20, n_static_obstacles=5, n_dynamic_obstacles=3):
        self.n_agents = n_agents
        self.world_size = world_size
        self.max_steps = 200
        self.current_step = 0
        self.n_static_obstacles = n_static_obstacles
        self.n_dynamic_obstacles = n_dynamic_obstacles
        self.agent_radius = 0.5
        self.obstacle_radius = 1.0

        # 状态空间: 每个智能体的位置(x,y)和速度(vx,vy)
        self.state_dim = 4

        # 动作空间: 每个智能体的加速度(ax,ay)
        self.action_dim = 2

        # 导航目标点
        self.target_position = np.array([self.world_size * 0.8, self.world_size * 0.8])

        # 基本队形配置 - 可用的队形模板
        self.formation_templates = {
            'line': self._generate_line_formation,
            'column': self._generate_column_formation,
            'triangle': self._generate_triangle_formation,
            'square': self._generate_square_formation,
            'circle': self._generate_circle_formation
        }

        # 当前使用的队形
        self.current_formation = 'triangle'

        # 初始化改进的奖励计算器
        self.reward_calculator = ImprovedRewardCalculator(self)

        # 初始化环境
        self.reset()

    def _generate_line_formation(self):
        """生成一字型队形，适合穿过狭窄通道"""
        positions = np.zeros((self.n_agents, 2))
        for i in range(self.n_agents):
            positions[i] = [i - (self.n_agents - 1) / 2, 0]
        return positions

    def _generate_column_formation(self):
        """生成纵队队形，适合穿过狭窄通道"""
        positions = np.zeros((self.n_agents, 2))
        for i in range(self.n_agents):
            positions[i] = [0, i - (self.n_agents - 1) / 2]
        return positions

    def _generate_triangle_formation(self):
        """生成三角队形，适合包围或防御"""
        positions = np.zeros((self.n_agents, 2))
        if self.n_agents >= 3:
            # 三角形的基本队形
            positions[0] = [0, 0.5]  # 顶点

            # 如果只有3个智能体
            if self.n_agents == 3:
                positions[1] = [-0.5, -0.5]  # 左下
                positions[2] = [0.5, -0.5]  # 右下
            else:
                # 均匀分布在底边
                base_agents = self.n_agents - 1
                for i in range(base_agents):
                    angle = math.pi * (3 / 4 + i * (1 / 2) / (base_agents - 1))
                    positions[i + 1] = [math.cos(angle), math.sin(angle)]
        else:
            # 如果智能体不足3个，退化为直线
            return self._generate_line_formation()

        return positions

    def _generate_square_formation(self):
        """生成方形队形，适合全方位警戒"""
        positions = np.zeros((self.n_agents, 2))

        if self.n_agents >= 4:
            corners = 4
            # 先放置四个角
            for i in range(min(corners, self.n_agents)):
                angle = i * 2 * math.pi / corners
                positions[i] = [math.cos(angle), math.sin(angle)]

            # 剩余的智能体均匀分布在边上
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
            # 如果智能体不足4个，退化为三角形或直线
            return self._generate_triangle_formation()

        return positions

    def _generate_circle_formation(self):
        """生成圆形队形，适合全方位防御"""
        positions = np.zeros((self.n_agents, 2))

        for i in range(self.n_agents):
            angle = i * 2 * math.pi / self.n_agents
            positions[i] = [math.cos(angle), math.sin(angle)]

        return positions

    def _initialize_mixed_static_obstacles(self):
        """初始化混合的静态障碍物 - 部分固定位置，部分随机生成"""
        # 固定障碍物的位置和大小
        fixed_obstacles = [
            # 中间的大障碍物
            {'position': np.array([10.0, 10.0]), 'radius': 2.0},

            # 左侧障碍物
            {'position': np.array([5.0, 7.0]), 'radius': 1.2},

            # 右侧障碍物
            {'position': np.array([15.0, 12.0]), 'radius': 1.3},

            # 靠近目标的障碍物
            {'position': np.array([14.0, 16.0]), 'radius': 0.8},
        ]

        # 确定固定障碍物的数量 - 保证不超过要求的总数
        n_fixed = min(len(fixed_obstacles), self.n_static_obstacles)
        obstacles = fixed_obstacles[:n_fixed]

        # 如果要求的障碍物数量大于固定障碍物数量，随机生成剩余的
        n_random = self.n_static_obstacles - n_fixed
        if n_random > 0:
            # 保存已有障碍物的位置，防止重叠
            existing_positions = [obs['position'] for obs in obstacles]
            if hasattr(self, 'positions'):
                existing_positions.extend([self.positions[i] for i in range(self.n_agents)])
            existing_positions.append(self.target_position)

            # 随机生成剩余的障碍物
            safe_radius_start = 3.0
            safe_radius_target = 3.0

            for _ in range(n_random):
                valid_position = False
                attempts = 0

                while not valid_position and attempts < 100:
                    # 随机位置
                    pos = np.random.rand(2) * self.world_size

                    # 检查是否与现有位置重叠
                    valid = True
                    for existing_pos in existing_positions:
                        if np.linalg.norm(pos - existing_pos) < safe_radius_start:
                            valid = False
                            break

                    # 检查是否与目标重叠
                    if np.linalg.norm(pos - self.target_position) < safe_radius_target:
                        valid = False

                    # 检查是否与其他随机生成的障碍物重叠
                    for obs in obstacles[n_fixed:]:
                        if np.linalg.norm(pos - obs['position']) < 2 * self.obstacle_radius:
                            valid = False
                            break

                    if valid:
                        valid_position = True
                        obstacles.append({
                            'position': pos,
                            'radius': self.obstacle_radius + np.random.rand() * 0.5  # 随机调整大小
                        })
                        existing_positions.append(pos)

                    attempts += 1

        return obstacles

    def _initialize_mixed_dynamic_obstacles(self):
        """初始化混合的动态障碍物 - 部分固定起点，部分随机生成"""
        # 固定起点的动态障碍物
        fixed_obstacles = [
            # 左上区域的动态障碍物
            {
                'position': np.array([8.0, 15.0]),
                'radius': 0.8,
                'velocity': np.array([0.05, -0.05])  # 向右下移动
            },

            # 右下区域的动态障碍物
            {
                'position': np.array([15.0, 8.0]),
                'radius': 0.8,
                'velocity': np.array([-0.05, 0.05])  # 向左上移动
            },
        ]

        # 确定固定障碍物的数量 - 保证不超过要求的总数
        n_fixed = min(len(fixed_obstacles), self.n_dynamic_obstacles)
        obstacles = fixed_obstacles[:n_fixed]

        # 如果要求的障碍物数量大于固定障碍物数量，随机生成剩余的
        n_random = self.n_dynamic_obstacles - n_fixed
        if n_random > 0:
            # 保存已有障碍物的位置
            existing_positions = [obs['position'] for obs in obstacles]
            if hasattr(self, 'static_obstacles'):
                existing_positions.extend([obs['position'] for obs in self.static_obstacles])
            if hasattr(self, 'positions'):
                existing_positions.extend([self.positions[i] for i in range(self.n_agents)])
            existing_positions.append(self.target_position)

            # 随机生成剩余的障碍物
            safe_radius_start = 3.0
            safe_radius_target = 3.0

            for _ in range(n_random):
                valid_position = False
                attempts = 0

                while not valid_position and attempts < 100:
                    # 随机位置
                    pos = np.random.rand(2) * self.world_size

                    # 检查是否与现有位置重叠
                    valid = True
                    for existing_pos in existing_positions:
                        if np.linalg.norm(pos - existing_pos) < safe_radius_start:
                            valid = False
                            break

                    # 检查是否与目标重叠
                    if np.linalg.norm(pos - self.target_position) < safe_radius_target:
                        valid = False

                    if valid:
                        valid_position = True
                        # 随机速度和方向
                        angle = np.random.rand() * 2 * np.pi
                        speed = 0.1 + np.random.rand() * 0.2  # 随机速度
                        velocity = np.array([np.cos(angle), np.sin(angle)]) * speed

                        obstacles.append({
                            'position': pos,
                            'radius': self.obstacle_radius,
                            'velocity': velocity
                        })
                        existing_positions.append(pos)

                    attempts += 1

        return obstacles

    def _update_dynamic_obstacles(self):
        """更新动态障碍物位置"""
        for obstacle in self.dynamic_obstacles:
            # 更新位置
            obstacle['position'] += obstacle['velocity']

            # 检查边界并反弹
            for i in range(2):
                if obstacle['position'][i] <= 0 or obstacle['position'][i] >= self.world_size:
                    obstacle['velocity'][i] *= -1
                    obstacle['position'][i] = np.clip(obstacle['position'][i], 0, self.world_size)

            # 随机改变方向 (小概率)
            if np.random.rand() < 0.02:
                angle = np.random.rand() * 2 * np.pi
                speed = np.linalg.norm(obstacle['velocity'])
                obstacle['velocity'] = np.array([np.cos(angle), np.sin(angle)]) * speed

    def _detect_nearby_obstacles(self, position, detection_radius=5.0):
        """检测给定位置附近的障碍物"""
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

    """根据环境障碍物动态调整队形"""
    def _adapt_formation(self):
        # 计算智能体当前中心
        center = np.mean(self.positions, axis=0)

        # 获取中心点附近的障碍物
        nearby_obstacles = self._detect_nearby_obstacles(center, detection_radius=6.0)

        # 如果附近没有障碍物，使用三角形队形(默认)
        if not nearby_obstacles:
            self.current_formation = 'triangle'
            return

        # 计算到目标的方向
        direction_to_target = self.target_position - center
        direction_to_target = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-10)

        # 分析障碍物分布
        obstacle_directions = []
        for obstacle in nearby_obstacles:
            # 从中心到障碍物的方向
            direction = obstacle['position'] - center
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            obstacle_directions.append(direction)

        # 统计各方向的障碍物数量
        front_count = 0
        side_count = 0
        for direction in obstacle_directions:
            # 计算障碍物方向与目标方向的夹角
            cos_angle = np.dot(direction, direction_to_target)
            if cos_angle > 0.7:  # 前方障碍物 (小于45度)
                front_count += 1
            elif abs(cos_angle) < 0.7:  # 侧方障碍物
                side_count += 1

        # 根据障碍物分布选择队形
        if front_count > 0 and side_count == 0:
            # 前方有障碍物，侧面无障碍物，使用线型队形垂直于前进方向
            self.current_formation = 'line'
            # 旋转队形使其垂直于前进方向
            perp_direction = np.array([-direction_to_target[1], direction_to_target[0]])
            self.formation_orientation = perp_direction
        elif front_count > 0 and side_count > 0:
            # 前方和侧面都有障碍物，使用紧凑队形
            self.current_formation = 'circle'
        elif side_count > 0 and front_count == 0:
            # 只有侧面有障碍物，使用纵队队形
            self.current_formation = 'column'
            self.formation_orientation = direction_to_target
        else:
            # 默认使用三角形队形
            self.current_formation = 'triangle'
            self.formation_orientation = direction_to_target

    """获取当前环境下期望的队形位置"""
    def _get_desired_formation_positions(self, scale=1.5):
        # 根据当前选择的队形生成相对位置
        formation_func = self.formation_templates[self.current_formation]
        relative_positions = formation_func()

        # 缩放队形
        relative_positions *= scale

        # 计算队伍中心
        center = np.mean(self.positions, axis=0)

        # 计算到目标的方向
        direction_to_target = self.target_position - center
        direction_to_target = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-10)

        # 根据前进方向旋转队形
        # 首先计算旋转矩阵，使队形的前方朝向目标
        if hasattr(self, 'formation_orientation'):
            forward = self.formation_orientation
        else:
            forward = direction_to_target

        # 默认队形的前方是y轴正方向[0,1]
        default_forward = np.array([0, 1])

        # 计算旋转角度
        cos_angle = np.dot(default_forward, forward)
        # 修复NumPy 2D向量叉积的警告，使用atan2来计算角度
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

        # 根据中心位置计算绝对位置
        desired_positions = center + rotated_positions

        return desired_positions

    def reset(self):
        """重置环境"""
        self.current_step = 0

        # 初始化智能体位置 - 靠近起点
        start_position = np.array([self.world_size * 0.2, self.world_size * 0.2])
        spread = 2.0  # 初始分散范围

        self.positions = np.zeros((self.n_agents, 2))
        for i in range(self.n_agents):
            self.positions[i] = start_position + (np.random.rand(2) * 2 - 1) * spread
            self.positions[i] = np.clip(self.positions[i], 0, self.world_size)

        # 初始速度为0
        self.velocities = np.zeros((self.n_agents, 2))

        # 初始化用于计算进展奖励的前一步距离
        team_center = np.mean(self.positions, axis=0)
        self.prev_distance_to_target = np.linalg.norm(team_center - self.target_position)

        # 使用混合障碍物方法
        self.static_obstacles = self._initialize_mixed_static_obstacles()
        self.dynamic_obstacles = self._initialize_mixed_dynamic_obstacles()

        # 初始化队形
        self.current_formation = 'triangle'  # 默认队形
        self._adapt_formation()

        # 返回每个智能体观察到的状态
        return self._get_observations()

    def step(self, actions):
        """执行动作并返回新状态、奖励等"""
        self.current_step += 1

        # 更新动态障碍物
        self._update_dynamic_obstacles()

        # 调整队形
        if self.current_step % 5 == 0:  # 每5步调整一次队形
            self._adapt_formation()

        # 初始化碰撞标志
        collision_occurred = False
        collision_type = None
        collision_agent = None

        # 更新速度和位置
        for i in range(self.n_agents):
            # 限制动作范围
            action = np.clip(actions[i], -1, 1)

            # 更新速度
            self.velocities[i] += action * 0.1  # 降低加速度以平滑移动

            # 限制速度大小
            speed = np.linalg.norm(self.velocities[i])
            max_speed = 0.5
            if speed > max_speed:
                self.velocities[i] = self.velocities[i] / speed * max_speed

            # 临时存储当前位置，用于检测碰撞
            old_position = self.positions[i].copy()

            # 更新位置
            self.positions[i] += self.velocities[i]

            # 确保在环境范围内
            self.positions[i] = np.clip(self.positions[i], 0, self.world_size)

            # 检测碰撞
            has_collision, collision_detail = self._check_collision(i)
            if has_collision:
                collision_occurred = True
                collision_type = collision_detail
                collision_agent = i
                break  # 发现碰撞立即停止

        # 计算奖励 - 使用改进的奖励计算器
        rewards = self.reward_calculator.compute_rewards()

        # 检查是否到达目标
        reached_target = self._check_target_reached()

        # 判断是否结束 - 到达目标或超过最大步数或发生碰撞
        done = reached_target or self.current_step >= self.max_steps or collision_occurred

        # 获取观察
        observations = self._get_observations()

        info = {
            'reached_target': reached_target,
            'current_formation': self.current_formation,
            'step': self.current_step,
            'collision_occurred': collision_occurred,
            'collision_type': collision_type,
            'collision_agent': collision_agent
        }

        return observations, rewards, done, info

    def _check_collision(self, agent_idx):
        """检查智能体是否与障碍物或其他智能体碰撞"""
        # 与静态障碍物碰撞检测
        for obstacle in self.static_obstacles:
            distance = np.linalg.norm(self.positions[agent_idx] - obstacle['position'])
            if distance < (self.agent_radius + obstacle['radius']):
                return True, 'static_obstacle'

        # 与动态障碍物碰撞检测
        for obstacle in self.dynamic_obstacles:
            distance = np.linalg.norm(self.positions[agent_idx] - obstacle['position'])
            if distance < (self.agent_radius + obstacle['radius']):
                return True, 'dynamic_obstacle'

        # 与其他智能体碰撞检测
        for i in range(self.n_agents):
            if i != agent_idx:
                distance = np.linalg.norm(self.positions[agent_idx] - self.positions[i])
                if distance < (2 * self.agent_radius):
                    return True, 'agent_collision'

        return False, None


    def _check_target_reached(self, threshold=2.0):
        """检查是否所有智能体都达到目标区域"""
        # 计算每个智能体到目标的距离
        distances = [np.linalg.norm(self.positions[i] - self.target_position) for i in range(self.n_agents)]

        # 如果超过80%的智能体到达目标区域，则认为任务成功
        success_threshold = 0.8 * self.n_agents
        agents_reached = sum(1 for d in distances if d < threshold)

        return agents_reached >= success_threshold

    def _get_observations(self):
        """获取每个智能体的观察"""
        observations = []

        desired_positions = self._get_desired_formation_positions()

        for i in range(self.n_agents):
            # 自身状态: 位置和速度
            own_state = np.concatenate([self.positions[i], self.velocities[i]])

            # 目标位置相对于自身的向量
            target_relative = self.target_position - self.positions[i]

            # 期望队形位置相对于自身的向量
            formation_relative = desired_positions[i] - self.positions[i]

            # 检测附近的障碍物
            nearby_obstacles = self._detect_nearby_obstacles(self.positions[i], detection_radius=4.0)

            # 将最近的3个障碍物信息添加到观察中
            obstacle_info = []
            sorted_obstacles = sorted(nearby_obstacles, key=lambda x: x['distance'])[:3]

            for obs in sorted_obstacles:
                # 障碍物相对位置
                rel_pos = obs['position'] - self.positions[i]
                # 对于动态障碍物，还包括其速度
                if obs['type'] == 'dynamic':
                    obstacle_info.extend([*rel_pos, *obs['velocity'], obs['radius']])
                else:
                    obstacle_info.extend([*rel_pos, 0, 0, obs['radius']])

            # 如果障碍物不足3个，填充零向量
            while len(sorted_obstacles) < 3:
                obstacle_info.extend([0, 0, 0, 0, 0])
                sorted_obstacles.append(None)

            # 其他智能体相对位置
            teammate_info = []
            for j in range(self.n_agents):
                if i != j:
                    rel_pos = self.positions[j] - self.positions[i]
                    rel_vel = self.velocities[j]
                    teammate_info.extend([*rel_pos, *rel_vel])

            # 组合所有观察
            observation = np.concatenate([
                own_state,  # 4 - 自身状态
                target_relative,  # 2 - 目标相对位置
                formation_relative,  # 2 - 期望队形位置
                np.array([int(self.current_formation == 'line'),
                          int(self.current_formation == 'column'),
                          int(self.current_formation == 'triangle'),
                          int(self.current_formation == 'square'),
                          int(self.current_formation == 'circle')]),  # 5 - 当前队形的独热编码
                np.array(obstacle_info),  # 15 - 3个障碍物信息
                np.array(teammate_info)  # 4*(n_agents-1) - 其他智能体信息
            ])

            observations.append(observation)

        return observations

    def render(self, mode='human'):
        """可视化当前环境状态"""
        fig = plt.figure(figsize=(10, 10))
        plt.xlim(0, self.world_size)
        plt.ylim(0, self.world_size)

        # 绘制目标位置
        plt.plot(self.target_position[0], self.target_position[1], 'x', color='red', markersize=15)
        plt.text(self.target_position[0], self.target_position[1] + 1, "Target", fontsize=12)

        # 绘制静态障碍物
        for obstacle in self.static_obstacles:
            circle = plt.Circle(obstacle['position'], obstacle['radius'], color='gray', alpha=0.7)
            plt.gcf().gca().add_artist(circle)

        # 绘制动态障碍物
        for obstacle in self.dynamic_obstacles:
            circle = plt.Circle(obstacle['position'], obstacle['radius'], color='orange', alpha=0.7)
            plt.gcf().gca().add_artist(circle)

            # 绘制速度向量
            plt.arrow(obstacle['position'][0], obstacle['position'][1],
                      obstacle['velocity'][0] * 3, obstacle['velocity'][1] * 3,
                      head_width=0.3, head_length=0.4, fc='orange', ec='orange', alpha=0.7)

        # 绘制智能体
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        # 首先绘制队形期望位置
        desired_positions = self._get_desired_formation_positions()
        for i in range(self.n_agents):
            circle = plt.Circle(desired_positions[i], self.agent_radius / 2,
                                color=colors[i % len(colors)], alpha=0.3)
            plt.gcf().gca().add_artist(circle)

        # 绘制智能体
        for i in range(self.n_agents):
            circle = plt.Circle(self.positions[i], self.agent_radius,
                                color=colors[i % len(colors)], alpha=0.7)
            plt.gcf().gca().add_artist(circle)

            # 绘制速度向量
            plt.arrow(self.positions[i][0], self.positions[i][1],
                      self.velocities[i][0] * 2, self.velocities[i][1] * 2,
                      head_width=0.2, head_length=0.3,
                      fc=colors[i % len(colors)], ec=colors[i % len(colors)])

            # 绘制编号
            plt.text(self.positions[i][0], self.positions[i][1], f'{i + 1}',
                     color='white', fontsize=8, ha='center', va='center')

        # 绘制队形类型和当前步骤
        plt.title(f'Step: {self.current_step} | Formation: {self.current_formation}')
        plt.grid(True)

        if mode == 'human':
            plt.savefig(f'formation_step_{self.current_step:03d}.png')
            plt.close(fig)  # 修复内存泄漏
            return None
        elif mode == 'return':
            return fig
        else:
            plt.close(fig)  # 确保关闭图形
            return None


# 改进的奖励计算器
class ImprovedRewardCalculator:
    def __init__(self, env):
        self.env = env
        self.reward_weights = {
            'progress': 5.0,
            'formation': 0.2,
            'collision': -3.0,
            'cooperation': 1.0,
            'efficiency': 0.5,
            'exploration': 0.1,
            'target_reach': 2.0,
            'survival': 0.01,
            'success': 10.0
        }

    def compute_rewards(self):
        """改进的奖励函数"""
        rewards = np.zeros(self.env.n_agents)

        # 获取期望的编队位置
        desired_positions = self.env._get_desired_formation_positions()

        # 队伍中心
        team_center = np.mean(self.env.positions, axis=0)

        # 计算到目标的距离
        distance_to_target = np.linalg.norm(team_center - self.env.target_position)
        prev_distance = getattr(self.env, 'prev_distance_to_target', distance_to_target)
        self.env.prev_distance_to_target = distance_to_target

        # 1. 进展奖励 - 鼓励向目标前进
        progress_reward = (prev_distance - distance_to_target) * self.reward_weights['progress']

        # 2. 合作奖励
        cooperation_reward = self._compute_cooperation_reward()

        # 3. 效率奖励
        efficiency_reward = self._compute_efficiency_reward()

        for i in range(self.env.n_agents):
            # 基础团队奖励
            rewards[i] += progress_reward
            rewards[i] += cooperation_reward
            rewards[i] += efficiency_reward

            # 1. 保持队形奖励
            formation_error = np.linalg.norm(self.env.positions[i] - desired_positions[i])
            formation_reward = -self.reward_weights['formation'] * formation_error
            rewards[i] += formation_reward

            # 4. 碰撞惩罚
            has_collision, _ = self.env._check_collision(i)
            if has_collision:
                rewards[i] += self.reward_weights['collision']

            # 5. 到达目标的额外奖励
            agent_distance_to_target = np.linalg.norm(self.env.positions[i] - self.env.target_position)
            if agent_distance_to_target < 2.0:
                rewards[i] += self.reward_weights['target_reach']

            # 6. 存活奖励
            rewards[i] += self.reward_weights['survival']

            # 7. 完成任务的大额奖励
            if self.env._check_target_reached():
                rewards[i] += self.reward_weights['success']

        return rewards


    def _compute_efficiency_reward(self):
        """计算效率奖励 - 鼓励直接路径"""
        team_center = np.mean(self.env.positions, axis=0)
        team_velocity = np.mean(self.env.velocities, axis=0)

        # 计算速度与目标方向的一致性
        direction_to_target = self.env.target_position - team_center
        direction_to_target = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-10)

        velocity_magnitude = np.linalg.norm(team_velocity)
        if velocity_magnitude > 1e-10:
            velocity_direction = team_velocity / velocity_magnitude
            alignment = np.dot(velocity_direction, direction_to_target)
            return max(0, alignment) * velocity_magnitude * self.reward_weights['efficiency']

        return 0


# 优化的高斯策略网络
class EfficientGaussianPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(EfficientGaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 使用更高效的网络结构
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # 比ReLU更平滑，收敛更快
            nn.LayerNorm(hidden_dim),  # 加速收敛
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # 共享特征提取
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # 权重初始化优化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # 对动作空间采样
        x_t = normal.rsample()  # reparameterization trick
        action = torch.tanh(x_t)

        # 计算对数概率
        log_prob = normal.log_prob(x_t)

        # 校正对数概率
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean

    def to(self, device):
        return super(EfficientGaussianPolicy, self).to(device)


# 优化的Q网络
class EfficientQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(EfficientQNetwork, self).__init__()

        # 第一个Q网络
        self.q1_net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        # 第二个Q网络 (用于减少过估计)
        self.q2_net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        # 输入处理
        batch_size = state.size(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if action.size(0) != batch_size:
            action = action.expand(batch_size, -1)

        xu = torch.cat([state, action], 1)

        # 计算两个Q值
        q1 = self.q1_net(xu)
        q2 = self.q2_net(xu)

        return q1, q2


# 修复的优先经验回放缓冲区
class PrioritizedReplayBuffer:
    def __init__(self, max_size=1e6, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.max_size = int(max_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(self.max_size, dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        # 新经验给予最高优先级
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None, None

        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # 重要性采样权重
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        # 增加beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # 重新组织样本
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*samples))

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch), indices, weights

    def update_priorities(self, indices, priorities):
        # 修复NumPy标量转换警告
        for idx, priority in zip(indices, priorities):
            # 确保priority是标量值
            if isinstance(priority, (np.ndarray, torch.Tensor)):
                if hasattr(priority, 'item'):
                    priority_val = priority.item()
                else:
                    priority_val = float(priority.flatten()[0])
            else:
                priority_val = float(priority)

            self.priorities[idx] = priority_val + 1e-6

    def __len__(self):
        return len(self.buffer)


# 完全重构的MASAC智能体 - 解决就地操作问题
class OptimizedMASACAgent:
    def __init__(self, input_dim, action_dim, hidden_dim=256, lr=3e-4, alpha=0.2, gamma=0.99, tau=0.005,
                 auto_entropy_tuning=True):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # 自动调整熵权重
        self.auto_entropy_tuning = auto_entropy_tuning

        # 使用优化的网络
        self.policy = EfficientGaussianPolicy(input_dim, action_dim, hidden_dim).to(device)
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)

        # 使用优化的Q网络
        self.critic = EfficientQNetwork(input_dim, action_dim, hidden_dim).to(device)
        self.critic_target = EfficientQNetwork(input_dim, action_dim, hidden_dim).to(device)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=1e-4)

        # 硬拷贝参数
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # 自动调整熵
        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr)

        # 梯度裁剪
        self.max_grad_norm = 0.5

        # 启用混合精度训练（如果支持）
        self.use_amp = torch.cuda.is_available() and hasattr(torch, 'amp')
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        # 禁用模型编译以避免Triton错误
        print("警告：由于Triton库缺失，已禁用模型编译")

    def act(self, state, evaluate=False):
        """根据状态选择动作"""
        state = torch.FloatTensor(state).to(device).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                # 评估模式使用均值动作
                _, _, action = self.policy.sample(state)
                return action.detach().cpu().numpy()[0]
            else:
                # 训练模式使用采样动作
                action, _, _ = self.policy.sample(state)
                return action.detach().cpu().numpy()[0]

    def update(self, state, action, reward, next_state, done, weights=None, updates=0):
        """完全重构的更新方法 - 彻底解决就地操作问题"""

        # 确保所有输入都是正确格式的张量
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)

        reward_np = np.array(reward)
        if reward_np.ndim == 1:
            reward_np = reward_np.reshape(-1, 1)
        reward = torch.FloatTensor(reward_np).to(device)

        next_state = torch.FloatTensor(next_state).to(device)

        done_np = np.array(done)
        if done_np.ndim == 1:
            done_np = done_np.reshape(-1, 1)
        done = torch.FloatTensor(done_np).to(device)

        # 重要性采样权重
        if weights is not None:
            weights = torch.FloatTensor(weights).to(device).unsqueeze(1)
        else:
            weights = torch.ones(state.size(0), 1).to(device)

        # 第一步：单独更新Critic网络
        td_errors, critic_loss = self._update_critic(state, action, reward, next_state, done, weights)

        # 第二步：根据延迟策略更新Policy网络
        policy_loss = 0.0
        if updates % 2 == 0:
            policy_loss = self._update_policy(state.detach().clone())  # 使用分离的状态副本

            # 更新熵权重
            if self.auto_entropy_tuning:
                self._update_alpha(state.detach().clone())  # 使用分离的状态副本

            # 软更新目标网络
            self._soft_update()

        # 返回损失值
        return td_errors.detach().cpu().numpy(), critic_loss, policy_loss

    def _update_critic(self, state, action, reward, next_state, done, weights):
        """单独更新Critic网络"""

        # 计算目标Q值
        with torch.no_grad():
            # 为下一状态采样动作
            next_action, next_log_pi, _ = self.policy.sample(next_state)
            next_q1, next_q2 = self.critic_target(next_state, next_action)
            next_q = torch.min(next_q1, next_q2)
            next_q = next_q - self.alpha * next_log_pi
            q_target = reward + (1 - done) * self.gamma * next_q

        # 计算当前Q值
        current_q1, current_q2 = self.critic(state, action)

        # 计算损失
        q1_loss = (weights * F.mse_loss(current_q1, q_target, reduction='none')).mean()
        q2_loss = (weights * F.mse_loss(current_q2, q_target, reduction='none')).mean()
        critic_loss = q1_loss + q2_loss

        # TD误差（用于更新优先级）
        td_errors = torch.abs(current_q1 - q_target).squeeze()

        # 更新Critic网络
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                self.critic_optimizer.zero_grad()
                self.scaler.scale(critic_loss).backward()
                self.scaler.unscale_(self.critic_optimizer)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.scaler.step(self.critic_optimizer)
                self.scaler.update()
        else:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        return td_errors, critic_loss.detach().item()

    def _update_policy(self, state):
        """单独更新Policy网络"""

        # 重新采样动作以避免计算图冲突
        pi, log_pi, _ = self.policy.sample(state)

        # 使用当前critic网络计算Q值
        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # 计算策略损失
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # 更新Policy网络
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()
                self.scaler.unscale_(self.policy_optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.scaler.step(self.policy_optimizer)
                self.scaler.update()
        else:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()

        return policy_loss.detach().item()

    def _update_alpha(self, state):
        """更新熵权重"""
        with torch.no_grad():
            # 重新采样以避免计算图冲突
            _, log_pi, _ = self.policy.sample(state)

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

    def _soft_update(self):
        """软更新目标网络"""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


# 优化的训练函数
def train_optimized_masac(env_class, n_episodes=2000, max_steps=200, batch_size=128, buffer_size=int(2e6),
                          print_every=10, render_every=100, save_every=200, comment='', num_parallel_envs=1,
                          use_prioritized_replay=True, **env_kwargs):
    """优化的MASAC训练函数"""
    # 创建TensorBoard写入器
    writer = create_tensorboard_writer(comment=f'OptimizedMASAC_{comment}')

    # 创建环境
    env = env_class(**env_kwargs)

    # 获取环境参数
    n_agents = env.n_agents

    # 获取观察空间维度
    test_obs = env.reset()
    obs_dim = len(test_obs[0])
    action_dim = env.action_dim

    print(f"观察空间维度: {obs_dim}, 动作空间维度: {action_dim}")

    # 初始化优化的智能体
    agents = [OptimizedMASACAgent(input_dim=obs_dim, action_dim=action_dim) for _ in range(n_agents)]

    # 初始化经验回放缓冲区
    if use_prioritized_replay:
        memory = PrioritizedReplayBuffer(max_size=buffer_size)
        print("使用优先经验回放")
    else:
        from collections import deque
        memory = deque(maxlen=int(buffer_size))
        print("使用简单缓冲区")

    # 训练指标
    episode_rewards = []
    success_rate = []
    formation_errors = []
    collision_counts = []

    # 模型存档路径
    os.makedirs('models', exist_ok=True)

    # 评估最佳模型
    best_success_rate = 0
    best_episode = 0

    # 总更新次数
    updates = 0

    # 开始训练
    for episode in range(1, n_episodes + 1):
        # 重置环境
        states = env.reset()

        episode_reward = 0
        collisions = 0

        # 每个智能体的损失
        critic_losses = [0] * n_agents
        actor_losses = [0] * n_agents
        update_counts = [0] * n_agents

        for step in range(max_steps):
            # 每个智能体选择动作
            actions = []
            for i, agent in enumerate(agents):
                action = agent.act(states[i])
                actions.append(action)

            # 执行动作
            next_states, rewards, done, info = env.step(actions)

            # 计算总奖励
            total_reward = sum(rewards)
            episode_reward += total_reward

            # 检测碰撞
            for i in range(n_agents):
                if env._check_collision(i):
                    collisions += 1

            # 存储经验
            for i in range(n_agents):
                if use_prioritized_replay:
                    memory.push(states[i], actions[i], rewards[i], next_states[i], done)
                else:
                    memory.append((states[i], actions[i], rewards[i], next_states[i], done))

            # 更新状态
            states = next_states

            # 从经验回放中采样并更新网络
            memory_size = len(memory)
            if memory_size > batch_size:
                for i, agent in enumerate(agents):
                    if use_prioritized_replay:
                        batch_data, indices, weights = memory.sample(batch_size)
                        if batch_data is not None:
                            state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch_data
                            td_errors, critic_loss, actor_loss = agent.update(
                                state_batch, action_batch, reward_batch, next_state_batch, done_batch,
                                weights, updates
                            )
                            # 更新优先级
                            memory.update_priorities(indices, td_errors)
                    else:
                        # 简单采样
                        batch = random.sample(list(memory), batch_size)
                        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array,
                                                                                                    zip(*batch))

                        # 确保正确的形状
                        if reward_batch.ndim == 1:
                            reward_batch = reward_batch.reshape(-1, 1)
                        if done_batch.ndim == 1:
                            done_batch = done_batch.reshape(-1, 1)

                        _, critic_loss, actor_loss = agent.update(
                            state_batch, action_batch, reward_batch, next_state_batch, done_batch,
                            None, updates
                        )

                    updates += 1

                    # 累计损失值
                    critic_losses[i] += critic_loss
                    actor_losses[i] += actor_loss
                    update_counts[i] += 1

            # 可视化
            if episode % render_every == 0 and step % 10 == 0:
                env.render()

            if done:
                break

        # 计算各项指标
        avg_critic_losses = [losses / max(1, count) for losses, count in zip(critic_losses, update_counts)]
        avg_actor_losses = [losses / max(1, count) for losses, count in zip(actor_losses, update_counts)]

        # 记录训练指标
        episode_rewards.append(episode_reward)
        current_success = 1 if info.get('reached_target', False) else 0
        success_rate.append(current_success)

        # 计算队形误差
        desired_positions = env._get_desired_formation_positions()
        formation_error = sum([np.linalg.norm(env.positions[i] - desired_positions[i]) for i in range(n_agents)])
        formation_errors.append(formation_error)

        collision_counts.append(collisions)

        # 记录到TensorBoard
        writer.add_scalar('Rewards/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Success/Success_Rate', current_success, episode)
        writer.add_scalar('Errors/Formation_Error', formation_error, episode)
        writer.add_scalar('Errors/Collision_Count', collisions, episode)
        writer.add_scalar('Training/Steps_Per_Episode', step + 1, episode)

        # 记录每个智能体的损失
        for i in range(n_agents):
            writer.add_scalar(f'Agent_{i}/Critic_Loss', avg_critic_losses[i], episode)
            writer.add_scalar(f'Agent_{i}/Actor_Loss', avg_actor_losses[i], episode)
            writer.add_scalar(f'Agent_{i}/Reward', rewards[i], episode)
            writer.add_scalar(f'Agent_{i}/Alpha', agents[i].alpha.detach().item(), episode)

            # 记录当前队形
        writer.add_text('Formation', info.get('current_formation', 'unknown'), episode)

        # 计算移动平均值
        window = min(print_every, episode)
        avg_reward = np.mean(episode_rewards[-window:])
        avg_success = np.mean(success_rate[-window:]) * 100
        avg_formation_error = np.mean(formation_errors[-window:])
        avg_collisions = np.mean(collision_counts[-window:])

        # 记录移动平均值
        writer.add_scalar('Rewards/Average_Reward', avg_reward, episode)
        writer.add_scalar('Success/Success_Rate_Average', avg_success, episode)
        writer.add_scalar('Errors/Average_Formation_Error', avg_formation_error, episode)
        writer.add_scalar('Errors/Average_Collision_Count', avg_collisions, episode)

        # 每隔一定回合，添加环境图像到TensorBoard
        if episode % render_every == 0:
            fig = env.render(mode='return')
            if fig is not None:
                writer.add_figure('Environment', fig, episode)
                plt.close(fig)

        # 输出训练进度
        if episode % print_every == 0:
            print(f"Episode {episode}/{n_episodes}")
            print(f"  平均奖励: {avg_reward:.2f}")
            print(f"  成功率: {avg_success:.1f}%")
            print(f"  平均队形误差: {avg_formation_error:.2f}")
            print(f"  平均碰撞次数: {avg_collisions:.2f}")
            print(f"  队形: {info.get('current_formation', 'unknown')}")
            print(f"  步数: {step + 1}/{max_steps}")
            print(f"  平均Critic损失: {np.mean(avg_critic_losses):.4f}")
            print(f"  平均Actor损失: {np.mean(avg_actor_losses):.4f}")
            print(f"  熵权重: {agents[0].alpha.detach().item():.4f}")
            print(f"  缓冲区大小: {len(memory)}")
            print()

        # 保存当前模型
        if episode % save_every == 0:
            save_dir = f'models/episode_{episode}'
            os.makedirs(save_dir, exist_ok=True)
            for i, agent in enumerate(agents):
                torch.save(agent.policy.state_dict(), f'{save_dir}/agent_{i}_policy.pth')
                torch.save(agent.critic.state_dict(), f'{save_dir}/agent_{i}_critic.pth')

        # 保存最佳模型
        recent_success_rate = np.mean(success_rate[-min(100, len(success_rate)):]) * 100
        if recent_success_rate > best_success_rate:
            best_success_rate = recent_success_rate
            best_episode = episode
            save_dir = 'models/best'
            os.makedirs(save_dir, exist_ok=True)
            for i, agent in enumerate(agents):
                torch.save(agent.policy.state_dict(), f'{save_dir}/agent_{i}_policy.pth')
                torch.save(agent.critic.state_dict(), f'{save_dir}/agent_{i}_critic.pth')
            print(f"🏆 保存最佳模型 (回合 {episode}), 成功率: {best_success_rate:.1f}%")

        # 早停检查（如果连续500个回合没有改进）
        if episode > 1000 and recent_success_rate > 80:
            print(f"🎯 达到满意的成功率 ({recent_success_rate:.1f}%)，提前结束训练")
            break

        # 保存最终模型
    save_dir = 'models/final'
    os.makedirs(save_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        torch.save(agent.policy.state_dict(), f'{save_dir}/agent_{i}_policy.pth')
        torch.save(agent.critic.state_dict(), f'{save_dir}/agent_{i}_critic.pth')

    print(f"✅ 训练完成! 最佳模型在回合 {best_episode} 的成功率: {best_success_rate:.1f}%")

    # 关闭TensorBoard写入器
    writer.close()

    # 返回训练历史和智能体
    return {
        'episode_rewards': episode_rewards,
        'success_rate': success_rate,
        'formation_errors': formation_errors,
        'collision_counts': collision_counts,
        'best_episode': best_episode,
        'best_success_rate': best_success_rate
    }, agents

    # 超参数调度器


class HyperparameterScheduler:
    def __init__(self):
        self.schedules = {
            'learning_rate': self._cosine_schedule,
            'exploration_noise': self._exponential_decay,
            'entropy_weight': self._adaptive_entropy
        }

    def _cosine_schedule(self, initial_lr, episode, total_episodes):
        return initial_lr * 0.5 * (1 + np.cos(np.pi * episode / total_episodes))

    def _exponential_decay(self, initial_noise, episode, decay_rate=0.995):
        return initial_noise * (decay_rate ** episode)

    def _adaptive_entropy(self, current_alpha, q_loss, target_range=(0.1, 0.5)):
        """根据Q损失自适应调整熵权重"""
        if q_loss > target_range[1]:
            return min(1.0, current_alpha * 1.1)  # 增加探索
        elif q_loss < target_range[0]:
            return max(0.01, current_alpha * 0.9)  # 减少探索
        return current_alpha

    # 训练监控器


class TrainingMonitor:
    def __init__(self, patience=500, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('-inf')
        self.wait = 0
        self.performance_history = deque(maxlen=100)

    def update(self, current_score):
        self.performance_history.append(current_score)

        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.wait = 0
            return False  # 不停止
        else:
            self.wait += 1

        return self.wait >= self.patience

    def get_recent_performance(self):
        if len(self.performance_history) == 0:
            return 0
        return np.mean(list(self.performance_history))

    # 测试智能体


def test_agents(env, agents, n_episodes=5, max_steps=200):
    """测试训练好的智能体"""
    success_count = 0
    total_rewards = []
    total_steps = []

    for episode in range(n_episodes):
        states = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 智能体选择动作
            actions = []
            for i, agent in enumerate(agents):
                action = agent.act(states[i], evaluate=True)  # 测试时使用确定性动作
                actions.append(action)

            # 执行动作
            next_states, rewards, done, info = env.step(actions)
            states = next_states
            episode_reward += sum(rewards)

            # 可视化
            if episode == 0:  # 只在第一个测试回合可视化
                env.render()

            if done:
                if info.get('reached_target', False):
                    success_count += 1
                total_steps.append(step + 1)
                break

        total_rewards.append(episode_reward)

    success_rate = success_count / n_episodes * 100
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps) if total_steps else max_steps

    print(f"📊 测试结果:")
    print(f"  成功率: {success_rate:.1f}%")
    print(f"  平均奖励: {avg_reward:.2f}")
    print(f"  平均步数: {avg_steps:.1f}")

    # 生成编队控制导航的动画


def create_formation_animation(env, agents, max_steps=200):
    """创建编队控制导航的动画"""
    # 重置环境
    states = env.reset()

    # 存储每一步的环境状态
    position_history = []
    velocity_history = []
    formation_history = []
    dynamic_obstacle_history = []

    for step in range(max_steps):
        # 记录当前状态
        position_history.append(env.positions.copy())
        velocity_history.append(env.velocities.copy())
        formation_history.append(env.current_formation)
        dynamic_obstacle_history.append([
            {'position': obs['position'].copy(), 'radius': obs['radius'], 'velocity': obs['velocity'].copy()}
            for obs in env.dynamic_obstacles
        ])

        # 智能体选择动作
        actions = []
        for i, agent in enumerate(agents):
            action = agent.act(states[i], evaluate=True)
            actions.append(action)

        # 执行动作
        next_states, rewards, done, info = env.step(actions)
        states = next_states

        if done:
            break

    if len(position_history) == 0:
        print("⚠️ 没有记录到任何状态，无法生成动画")
        return

    # 创建动画
    fig, ax = plt.subplots(figsize=(10, 10))

    # 设置边界
    ax.set_xlim(0, env.world_size)
    ax.set_ylim(0, env.world_size)

    # 绘制静态障碍物
    for obstacle in env.static_obstacles:
        circle = plt.Circle(obstacle['position'], obstacle['radius'], color='gray', alpha=0.7)
        ax.add_artist(circle)

    # 绘制目标位置
    target = plt.plot(env.target_position[0], env.target_position[1], 'x', color='red', markersize=15)[0]
    ax.text(env.target_position[0], env.target_position[1] + 1, "Target", fontsize=12)

    # 初始化动态障碍物
    dynamic_obstacles = []
    for _ in env.dynamic_obstacles:
        obstacle = plt.Circle((0, 0), 0, color='orange', alpha=0.7)
        ax.add_artist(obstacle)
        dynamic_obstacles.append(obstacle)

    # 初始化智能体
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    agents_circles = []
    agents_arrows = []

    for i in range(env.n_agents):
        agent_circle = plt.Circle((0, 0), env.agent_radius, color=colors[i % len(colors)], alpha=0.7)
        ax.add_artist(agent_circle)
        agents_circles.append(agent_circle)

        agent_arrow = ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.3, fc=colors[i % len(colors)],
                               ec=colors[i % len(colors)])
        agents_arrows.append(agent_arrow)

    # 添加文本标签
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    formation_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

    def update(frame):
        # 更新动态障碍物
        for i, obstacle in enumerate(dynamic_obstacles):
            if i < len(dynamic_obstacle_history[frame]):
                obs_data = dynamic_obstacle_history[frame][i]
                obstacle.center = obs_data['position']
                obstacle.radius = obs_data['radius']

        # 更新智能体
        for i in range(env.n_agents):
            # 更新位置
            agents_circles[i].center = position_history[frame][i]

            # 移除旧的箭头，创建新的箭头
            agents_arrows[i].remove()
            agents_arrows[i] = ax.arrow(
                position_history[frame][i][0],
                position_history[frame][i][1],
                velocity_history[frame][i][0] * 2,
                velocity_history[frame][i][1] * 2,
                head_width=0.2,
                head_length=0.3,
                fc=colors[i % len(colors)],
                ec=colors[i % len(colors)]
            )

        # 更新文本
        step_text.set_text(f'Step: {frame + 1}/{len(position_history)}')
        formation_text.set_text(f'Formation: {formation_history[frame]}')

        return dynamic_obstacles + agents_circles + agents_arrows + [step_text, formation_text]

    # 创建动画
    try:
        ani = FuncAnimation(fig, update, frames=len(position_history), interval=50, blit=True)
        ani.save('formation_navigation.gif', writer='pillow', fps=10)
        print("🎬 动画已保存为 'formation_navigation.gif'")
    except Exception as e:
        print(f"⚠️ 动画保存失败: {e}")
    finally:
        plt.close(fig)

    # 可视化训练历史


def visualize_training_history(history):
    """可视化训练历史"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 绘制奖励
    axs[0, 0].plot(history['episode_rewards'])
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].grid(True)

    # 绘制成功率
    window_size = min(100, len(history['success_rate']))
    if window_size > 0:
        success_rate_moving_avg = np.convolve(history['success_rate'],
                                              np.ones(window_size) / window_size,
                                              mode='valid')
        axs[0, 1].plot(success_rate_moving_avg)
    axs[0, 1].set_title('Success Rate (Moving Average)')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Success Rate')
    axs[0, 1].set_ylim([0, 1])
    axs[0, 1].grid(True)

    # 绘制队形误差
    axs[1, 0].plot(history['formation_errors'])
    axs[1, 0].set_title('Formation Error')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Error')
    axs[1, 0].grid(True)

    # 绘制碰撞次数
    axs[1, 1].plot(history['collision_counts'])
    axs[1, 1].set_title('Collision Count')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Collisions')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("📈 训练历史已保存为 'training_history.png'")

    # 加载训练好的模型


def load_trained_agents(model_path, n_agents, obs_dim, action_dim):
    """加载训练好的智能体"""
    agents = [OptimizedMASACAgent(input_dim=obs_dim, action_dim=action_dim) for _ in range(n_agents)]

    for i, agent in enumerate(agents):
        try:
            agent.policy.load_state_dict(torch.load(f'{model_path}/agent_{i}_policy.pth', map_location=device))
            agent.critic.load_state_dict(torch.load(f'{model_path}/agent_{i}_critic.pth', map_location=device))
            print(f"✅ 智能体 {i} 模型加载成功")
        except FileNotFoundError:
            print(f"❌ 智能体 {i} 模型文件未找到")
        except Exception as e:
            print(f"❌ 智能体 {i} 模型加载失败: {e}")

    return agents


if __name__ == "__main__":
    # 设置随机种子以保证可重现性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)

    print("🚀 开始优化的多智能体强化学习训练")
    print(f"设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"当前时间: {datetime.now()}")

    # 创建环境 - 使用3个智能体和混合障碍物
    try:
        env = AdaptiveFormationEnvironment(
            n_agents=3,
            world_size=20,
            n_static_obstacles=4,  # 使用4个静态障碍物（包括固定和随机）
            n_dynamic_obstacles=1  # 使用1个动态障碍物
        )

        print(f"环境创建成功: {env.n_agents}个智能体")
        print(f"静态障碍物: {len(env.static_obstacles)}个")
        print(f"动态障碍物: {len(env.dynamic_obstacles)}个")

        # 训练智能体，使用优化配置
        history, agents = train_optimized_masac(
            env_class=AdaptiveFormationEnvironment,
            n_agents=3,
            world_size=20,
            n_static_obstacles=4,
            n_dynamic_obstacles=1,
            n_episodes=3000,  # 适中的训练回合数
            max_steps=200,
            batch_size=128,  # 适中的批次大小以节省内存
            buffer_size=int(5e5),  # 减少缓冲区大小以节省内存
            print_every=50,
            render_every=500,  # 减少渲染频率以节省时间
            save_every=500,
            comment='completely_fixed_gradient_issues',
            num_parallel_envs=1,
            use_prioritized_replay=True  # 启用优先经验回放
        )

        # 可视化训练历史
        print("\n📊 生成训练历史图表...")
        visualize_training_history(history)

        # 测试智能体
        print("\n🧪 开始测试训练好的智能体...")
        test_agents(env, agents, n_episodes=5)

        # 创建动画
        print("\n🎬 生成导航动画...")
        create_formation_animation(env, agents)

        print("\n🎉 所有任务完成!")
        print(f"最佳模型成功率: {history['best_success_rate']:.1f}%")
        print(f"最佳模型回合: {history['best_episode']}")
        print(f"总训练回合: {len(history['episode_rewards'])}")

        # 保存训练配置信息
        config_info = {
            'training_episodes': len(history['episode_rewards']),
            'best_success_rate': history['best_success_rate'],
            'best_episode': history['best_episode'],
            'final_success_rate': np.mean(history['success_rate'][-100:]) * 100,
            'device': str(device),
            'pytorch_version': torch.__version__,
            'timestamp': datetime.now().isoformat()
        }

        import json

        with open('training_config.json', 'w') as f:
            json.dump(config_info, f, indent=2)

        print(f"📄 训练配置已保存到 'training_config.json'")

    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback

        traceback.print_exc()

        # 启用梯度异常检测进行调试
        print("\n🔍 启用梯度异常检测重新运行...")
        torch.autograd.set_detect_anomaly(True)

        try:
            # 重新尝试一个简化的训练过程
            env = AdaptiveFormationEnvironment(n_agents=3, world_size=20, n_static_obstacles=2, n_dynamic_obstacles=0)
            history, agents = train_optimized_masac(
                env_class=AdaptiveFormationEnvironment,
                n_agents=3, world_size=20, n_static_obstacles=2, n_dynamic_obstacles=0,
                n_episodes=100, max_steps=50, batch_size=32, buffer_size=int(1e4),
                print_every=10, render_every=100, save_every=50,
                comment='debug_mode', use_prioritized_replay=False
            )
            print("✅ 调试模式训练成功!")
        except Exception as debug_e:
            print(f"❌ 调试模式也失败: {debug_e}")
            traceback.print_exc()
