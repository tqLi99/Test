import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
import math
import random
import os
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def _initialize_static_obstacles(self):
        """初始化静态障碍物"""
        obstacles = []

        # 确保目标区域和起始区域没有障碍物
        safe_radius_start = 3.0
        safe_radius_target = 3.0

        for _ in range(self.n_static_obstacles):
            valid_position = False
            attempts = 0

            while not valid_position and attempts < 100:
                # 随机位置
                pos = np.random.rand(2) * self.world_size

                # 检查是否与起始区域和目标区域重叠
                dist_to_start = np.min([np.linalg.norm(pos - self.positions[i]) for i in range(self.n_agents)])
                dist_to_target = np.linalg.norm(pos - self.target_position)

                # 检查是否与其他障碍物重叠
                dist_to_obstacles = float('inf')
                if obstacles:
                    dist_to_obstacles = np.min([np.linalg.norm(pos - obs['position']) for obs in obstacles])

                if (dist_to_start > safe_radius_start and
                        dist_to_target > safe_radius_target and
                        dist_to_obstacles > 2 * self.obstacle_radius):
                    valid_position = True
                    obstacles.append({
                        'position': pos,
                        'radius': self.obstacle_radius + np.random.rand() * 0.5  # 随机调整大小
                    })

                attempts += 1

        return obstacles

    def _initialize_dynamic_obstacles(self):
        """初始化动态障碍物"""
        obstacles = []

        # 确保目标区域和起始区域没有障碍物
        safe_radius_start = 3.0
        safe_radius_target = 3.0

        for _ in range(self.n_dynamic_obstacles):
            valid_position = False
            attempts = 0

            while not valid_position and attempts < 100:
                # 随机位置
                pos = np.random.rand(2) * self.world_size

                # 检查是否与起始区域和目标区域重叠
                dist_to_start = np.min([np.linalg.norm(pos - self.positions[i]) for i in range(self.n_agents)])
                dist_to_target = np.linalg.norm(pos - self.target_position)

                # 检查是否与其他障碍物重叠
                dist_to_static = float('inf')
                if self.static_obstacles:
                    dist_to_static = np.min([np.linalg.norm(pos - obs['position']) for obs in self.static_obstacles])

                dist_to_dynamic = float('inf')
                if obstacles:
                    dist_to_dynamic = np.min([np.linalg.norm(pos - obs['position']) for obs in obstacles])

                if (dist_to_start > safe_radius_start and
                        dist_to_target > safe_radius_target and
                        dist_to_static > 2 * self.obstacle_radius and
                        dist_to_dynamic > 2 * self.obstacle_radius):
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

                attempts += 1

        return obstacles

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
            existing_positions.extend([obs['position'] for obs in self.static_obstacles])
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

    def _adapt_formation(self):
        """根据环境障碍物动态调整队形"""
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

    def _get_desired_formation_positions(self, scale=1.5):
        """获取当前环境下期望的队形位置"""
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
        sin_angle = np.cross(default_forward, forward)

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
            if self._check_collision(i):
                # 发生碰撞，恢复位置并反弹
                self.positions[i] = old_position
                self.velocities[i] *= -0.5  # 碰撞后反向，并减速

        # 计算奖励
        rewards = self._compute_rewards()

        # 检查是否到达目标
        reached_target = self._check_target_reached()

        # 判断是否结束 - 到达目标或超过最大步数
        done = reached_target or self.current_step >= self.max_steps

        # 获取观察
        observations = self._get_observations()

        info = {
            'reached_target': reached_target,
            'current_formation': self.current_formation,
            'step': self.current_step
        }

        return observations, rewards, done, info

    def _check_collision(self, agent_idx):
        """检查智能体是否与障碍物或其他智能体碰撞"""
        # 与静态障碍物碰撞检测
        for obstacle in self.static_obstacles:
            distance = np.linalg.norm(self.positions[agent_idx] - obstacle['position'])
            if distance < (self.agent_radius + obstacle['radius']):
                return True

        # 与动态障碍物碰撞检测
        for obstacle in self.dynamic_obstacles:
            distance = np.linalg.norm(self.positions[agent_idx] - obstacle['position'])
            if distance < (self.agent_radius + obstacle['radius']):
                return True

        # 与其他智能体碰撞检测
        for i in range(self.n_agents):
            if i != agent_idx:
                distance = np.linalg.norm(self.positions[agent_idx] - self.positions[i])
                if distance < (2 * self.agent_radius):
                    return True

        return False

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

    # 改进的奖励函数
    def _compute_rewards(self):
        """改进的奖励函数"""
        rewards = np.zeros(self.n_agents)

        # 获取期望的编队位置
        desired_positions = self._get_desired_formation_positions()

        # 队伍中心
        team_center = np.mean(self.positions, axis=0)

        # 计算到目标的距离
        distance_to_target = np.linalg.norm(team_center - self.target_position)
        prev_distance = getattr(self, 'prev_distance_to_target', distance_to_target)
        self.prev_distance_to_target = distance_to_target

        # 进展奖励 - 鼓励向目标前进
        progress_reward = (prev_distance - distance_to_target) * 5.0

        for i in range(self.n_agents):
            # 1. 保持队形奖励 - 使用较小权重
            formation_error = np.linalg.norm(self.positions[i] - desired_positions[i])
            formation_reward = -0.1 * formation_error  # 降低权重

            # 2. 进展奖励 - 基于团队距离目标的变化
            rewards[i] += progress_reward

            # 3. 队形保持奖励
            rewards[i] += formation_reward

            # 4. 碰撞惩罚 - 增加权重
            if self._check_collision(i):
                rewards[i] -= 2.0  # 增加惩罚

            # 5. 到达目标的额外奖励
            agent_distance_to_target = np.linalg.norm(self.positions[i] - self.target_position)
            if agent_distance_to_target < 2.0:
                rewards[i] += 2.0  # 增加奖励

            # 6. 存活奖励 - 鼓励智能体继续"存活"并尝试完成任务
            rewards[i] += 0.01

            # 7. 完成任务的大额奖励
            if self._check_target_reached():
                rewards[i] += 10.0  # 增加成功奖励

        return rewards

    def render(self, mode='human'):
        """可视化当前环境状态"""
        plt.figure(figsize=(10, 10))
        plt.xlim(0, self.world_size)
        plt.ylim(0, self.world_size)

        # 绘制目标位置
        plt.plot(self.target_position[0], self.target_position[1], 'x', color='red', markersize=15)
        plt.text(self.target_position[0], self.target_position[1] + 1, "Target", fontsize=12)  # 使用英文避免字体问题

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
        plt.title(f'Step: {self.current_step} | Formation: {self.current_formation}')  # 使用英文避免字体问题
        plt.grid(True)

        if mode == 'human':
            plt.savefig(f'formation_step_{self.current_step:03d}.png')
            plt.close()
            return None
        elif mode == 'return':
            return plt.gcf()
        else:
            return plt.gcf()


# MASAC的Actor网络 - 替代原MADDPG的Actor
class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 共享网络层
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # 均值和标准差
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        # 初始化
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
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
        return super(GaussianPolicy, self).to(device)


# MASAC的Q网络 - 替代原MADDPG的Critic
class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()

        # 第一个Q网络
        self.linear1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # 第二个Q网络 (用于减少过估计)
        self.linear4 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        # 初始化
        self.apply(weights_init_)

    def forward(self, state, action):
        # 输入处理
        batch_size = state.size(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if action.size(0) != batch_size:
            action = action.expand(batch_size, -1)

        xu = torch.cat([state, action], 1)

        # 第一个Q值
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # 第二个Q值
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


# 权重初始化函数
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# MASAC智能体 - 替代原MADDPG智能体
class MASACAgent:
    def __init__(self, input_dim, action_dim, hidden_dim=256, lr=3e-4, alpha=0.2, gamma=0.99, tau=0.005,
                 auto_entropy_tuning=True):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # 自动调整熵权重
        self.auto_entropy_tuning = auto_entropy_tuning

        # 策略网络
        self.policy = GaussianPolicy(input_dim, action_dim, hidden_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Q网络
        self.critic = QNetwork(input_dim, action_dim, hidden_dim).to(device)
        self.critic_target = QNetwork(input_dim, action_dim, hidden_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 硬拷贝参数
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # 自动调整熵
        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # 梯度裁剪
        self.max_grad_norm = 0.5

    def act(self, state, evaluate=False):
        """根据状态选择动作"""
        state = torch.FloatTensor(state).to(device).unsqueeze(0)

        if evaluate:
            # 评估模式使用均值动作
            _, _, action = self.policy.sample(state)
            return action.detach().cpu().numpy()[0]
        else:
            # 训练模式使用采样动作
            action, _, _ = self.policy.sample(state)
            return action.detach().cpu().numpy()[0]

    def update(self, state, action, reward, next_state, done, updates=0):
        """更新智能体的策略和价值函数"""
        # 确保所有输入都是张量并具有正确的形状
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

        # 更新Q网络
        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(next_state)
            next_q1, next_q2 = self.critic_target(next_state, next_action)
            next_q = torch.min(next_q1, next_q2)
            next_q = next_q - self.alpha * next_log_pi
            q_target = reward + (1 - done) * self.gamma * next_q

        # 当前Q值
        current_q1, current_q2 = self.critic(state, action)

        # Q网络损失
        q1_loss = F.mse_loss(current_q1, q_target)
        q2_loss = F.mse_loss(current_q2, q_target)
        critic_loss = q1_loss + q2_loss

        # 更新Q网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # 延迟更新策略网络
        if updates % 2 == 0:
            # 更新策略网络
            pi, log_pi, _ = self.policy.sample(state)
            qf1_pi, qf2_pi = self.critic(state, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()

            # 自动调整熵
            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp()

            # 软更新目标网络
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

            return critic_loss.item(), policy_loss.item()
        else:
            return critic_loss.item(), 0


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = []
        self.max_size = max_size
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        # 确保reward和done是列向量形式 [batch_size, 1]
        if reward.ndim == 1:
            reward = reward.reshape(-1, 1)
        if done.ndim == 1:
            done = done.reshape(-1, 1)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# 使用MASAC训练智能体
def train_masac(env, n_episodes=2000, max_steps=200, batch_size=128, buffer_size=int(1e6),
                print_every=10, render_every=100, save_every=200, comment=''):
    """训练MASAC智能体"""
    # 创建TensorBoard写入器
    writer = create_tensorboard_writer(comment=f'MASAC_{env.n_agents}agents_{comment}')

    # 获取环境参数
    n_agents = env.n_agents

    # 获取观察空间维度
    test_obs = env.reset()
    obs_dim = len(test_obs[0])
    action_dim = env.action_dim

    print(f"观察空间维度: {obs_dim}, 动作空间维度: {action_dim}")

    # 初始化智能体
    agents = [MASACAgent(input_dim=obs_dim, action_dim=action_dim) for _ in range(n_agents)]

    # 初始化经验回放缓冲区
    memory = ReplayBuffer(max_size=buffer_size)

    # 训练指标
    episode_rewards = []
    success_rate = []  # 成功率
    formation_errors = []  # 队形误差
    collision_counts = []  # 碰撞次数

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
                memory.push(states[i], actions[i], rewards[i], next_states[i], done)

            # 更新状态
            states = next_states

            # 从经验回放中采样并更新网络
            if len(memory) > batch_size:
                for i, agent in enumerate(agents):
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)
                    critic_loss, actor_loss = agent.update(
                        state_batch, action_batch, reward_batch, next_state_batch, done_batch, updates
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
            writer.add_scalar(f'Agent_{i}/Alpha', agents[i].alpha.item(), episode)

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
            writer.add_figure('Environment', fig, episode)

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
            print(f"  熵权重: {agents[0].alpha.item():.4f}")
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
            print(f"保存最佳模型 (回合 {episode}), 成功率: {best_success_rate:.1f}%")

    # 保存最终模型
    save_dir = 'models/final'
    os.makedirs(save_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        torch.save(agent.policy.state_dict(), f'{save_dir}/agent_{i}_policy.pth')
        torch.save(agent.critic.state_dict(), f'{save_dir}/agent_{i}_critic.pth')

    print(f"训练完成! 最佳模型在回合 {best_episode} 的成功率: {best_success_rate:.1f}%")

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


# 测试智能体
def test_agents(env, agents, n_episodes=5, max_steps=200):
    """测试训练好的智能体"""
    success_count = 0

    for episode in range(n_episodes):
        states = env.reset()

        for step in range(max_steps):
            # 智能体选择动作
            actions = []
            for i, agent in enumerate(agents):
                action = agent.act(states[i], evaluate=True)  # 测试时使用确定性动作
                actions.append(action)

            # 执行动作
            next_states, rewards, done, info = env.step(actions)
            states = next_states

            # 可视化
            env.render()

            if done:
                if info.get('reached_target', False):
                    success_count += 1
                break

    success_rate = success_count / n_episodes * 100
    print(f"测试成功率: {success_rate:.1f}%")



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
            action = agent.act(states[i], noise=False)
            actions.append(action)

        # 执行动作
        next_states, rewards, done, info = env.step(actions)
        states = next_states

        if done:
            break

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
    ani = FuncAnimation(fig, update, frames=len(position_history), interval=50, blit=True)

    # 保存为GIF
    ani.save('formation_navigation.gif', writer='pillow', fps=10)

    plt.close()
    print("动画已保存为 'formation_navigation.gif'")


# 可视化训练历史
def visualize_training_history(history):
    """可视化训练历史"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 绘制奖励
    axs[0, 0].plot(history['episode_rewards'])
    axs[0, 0].set_title('Average Reward')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].grid(True)

    # 绘制成功率
    window_size = min(100, len(history['success_rate']))
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
    plt.savefig('training_history.png')
    plt.close()


if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)

    # 创建环境 - 使用3个智能体和混合障碍物
    env = AdaptiveFormationEnvironment(n_agents=3, world_size=20,
                                       n_static_obstacles=4,  # 使用4个静态障碍物（包括固定和随机）
                                       n_dynamic_obstacles=1)  # 使用1个动态障碍物

    # 训练智能体，添加自定义注释
    history, agents = train_masac(env, n_episodes=20000, max_steps=200,
                                  comment='three_agents_mixed_obstacles_sac')

    # 可视化训练历史
    visualize_training_history(history)

    # 测试智能体
    test_agents(env, agents, n_episodes=5)

    # 创建动画
    create_formation_animation(env, agents)
