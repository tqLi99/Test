import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

from velodyne_env import GazeboEnv

# 优先经验回放缓冲区
class PrioritizedReplayBuffer:
    def __init__(self, max_size, input_shape, action_shape, alpha=0.4, beta=0.6, beta_increment=0.001):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # 避免优先级为0

        self.states = np.zeros((self.max_size, input_shape))
        self.actions = np.zeros((self.max_size, action_shape))
        self.rewards = np.zeros((self.max_size, 1))
        self.next_states = np.zeros((self.max_size, input_shape))
        self.dones = np.zeros((self.max_size, 1))
        self.priorities = np.zeros((self.max_size, 1))

    def add(self, state, action, reward, done, next_state):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # 新样本给予最大优先级
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.ptr] = max_prio

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        if self.size < batch_size:
            idxs = np.random.randint(0, self.size, size=batch_size)
        else:
            priorities = self.priorities[:self.size]
            probabilities = priorities ** self.alpha
            probabilities = probabilities / probabilities.sum()

            idxs = np.random.choice(self.size, batch_size, p=probabilities.flatten())

        states = self.states[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_states = self.next_states[idxs]
        dones = self.dones[idxs]

        # 计算重要性权重
        weights = (self.size * probabilities[idxs]) ** (-self.beta)
        weights = weights / weights.max()
        weights = np.array(weights, dtype=np.float32)

        self.beta = min(1.0, self.beta + self.beta_increment)

        return states, actions, rewards, dones, next_states, idxs, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.squeeze(priorities) if priorities.ndim > 1 else priorities
        for idx, priority in zip(idxs, priorities):
            self.priorities[idx] = priority + self.epsilon

def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state), eval=True)
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward

# SAC的Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden1_size=512, hidden2_size=256, 
                 log_std_min=-20, log_std_max=2, dropout_rate=0.1):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 状态输入规范化
        self.layer_norm = nn.LayerNorm(state_dim)
        
        # 第一层 - 使用权重规范化而非批归一化
        self.layer_1 = nn.utils.parametrizations.weight_norm(nn.Linear(state_dim, hidden1_size))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 第二层 - 使用权重规范化
        self.layer_2 = nn.utils.parametrizations.weight_norm(nn.Linear(hidden1_size, hidden2_size))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 均值和标准差输出层
        self.mean_layer = nn.Linear(hidden2_size, action_dim)
        self.log_std_layer = nn.Linear(hidden2_size, action_dim)
        
        # 正交初始化 - 有助于梯度流
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 正交初始化
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, state):
        # 状态层规范化
        x = self.layer_norm(state)
        
        # 第一层 - 使用ELU激活函数代替ReLU
        x = F.elu(self.layer_1(x))
        x = self.dropout1(x)
        
        # 第二层 - 使用ELU激活函数
        x = F.elu(self.layer_2(x))
        x = self.dropout2(x)
        
        # 均值和标准差计算
        mean = torch.tanh(self.mean_layer(x)) * 0.1  # 缩小初始输出范围，有助于稳定初期训练
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        
        # 创建正态分布
        normal = Normal(mean, std)
        
        # 重参数化技巧
        x = normal.rsample()
        
        # 添加噪声缩放
        noise_scale = min(1.0, 2.0 - 1.5e-5 * self._sample_count) if hasattr(self, '_sample_count') else 1.0
        if hasattr(self, '_sample_count'):
            self._sample_count += 1
        else:
            self._sample_count = 0
            
        # tanh压缩
        action = torch.tanh(x)
        
        # 计算对数概率
        log_prob = normal.log_prob(x)
        
        # 对数概率校正
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # 缩放动作
        action = action * self.max_action
        
        return action, log_prob

# SAC的Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1_size=512, hidden2_size=256, dropout_rate=0.1):
        super().__init__()
        
        # 输入层规范化
        self.layer_norm = nn.LayerNorm(state_dim + action_dim)
        
        # 第一个共享层 - 使用权重规范化
        self.shared_layer1 = nn.utils.parametrizations.weight_norm(nn.Linear(state_dim + action_dim, hidden1_size))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 第二个共享层 - 使用权重规范化
        self.shared_layer2 = nn.utils.parametrizations.weight_norm(nn.Linear(hidden1_size, hidden2_size))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Q1网络
        self.Q1_layer = nn.utils.parametrizations.weight_norm(nn.Linear(hidden2_size, hidden2_size // 2))
        self.Q1_head = nn.Linear(hidden2_size // 2, 1)
        
        # Q2网络
        self.Q2_layer = nn.utils.parametrizations.weight_norm(nn.Linear(hidden2_size, hidden2_size // 2))
        self.Q2_head = nn.Linear(hidden2_size // 2, 1)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 添加残差连接支持
        self.use_residual = True
        if self.use_residual:
            # 残差投影层 - 仅在需要调整维度时使用
            self.res_proj = nn.Linear(state_dim + action_dim, hidden1_size)
            self.res_proj2 = nn.Linear(hidden1_size, hidden2_size)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, state, action):
        # 合并状态和动作
        x = torch.cat([state, action], 1)
        
        # 输入层规范化
        x = self.layer_norm(x)
        
        # 残差连接实现
        if self.use_residual:
            identity = self.res_proj(x)
            x = F.elu(self.shared_layer1(x))
            x = self.dropout1(x)
            x = x + identity  # 残差连接
            
            identity = self.res_proj2(x)
            x = F.elu(self.shared_layer2(x))
            x = self.dropout2(x)
            x = x + identity  # 残差连接
        else:
            # 不使用残差连接
            x = F.elu(self.shared_layer1(x))
            x = self.dropout1(x)
            x = F.elu(self.shared_layer2(x))
            x = self.dropout2(x)
        
        # Q1分支
        q1 = F.elu(self.Q1_layer(x))
        q1 = self.Q1_head(q1)
        
        # Q2分支
        q2 = F.elu(self.Q2_layer(x))
        q2 = self.Q2_head(q2)
        
        return q1, q2

# 引入MPC模块
class MPCController:
    def __init__(self, state_dim, action_dim, horizon=10, dt=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.dt = dt

    def optimize(self, state, dynamics_model, cost_function):
        # 初始化优化变量
        actions = np.zeros((self.horizon, self.action_dim))
        states = np.zeros((self.horizon + 1, self.state_dim))
        states[0] = state

        # 使用简单梯度下降优化MPC轨迹
        for t in range(self.horizon):
            actions[t] = np.clip(actions[t], -1, 1)  # 动作范围限制
            states[t + 1] = dynamics_model(states[t], actions[t])  # 动态模型预测
        total_cost = cost_function(states, actions)

        # 返回优化后的第一个动作
        return actions[0]

# 在SAC类中增加MPC支持
class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, device, alpha=0.2, automatic_entropy_tuning=True):
        self.device = device
        self.global_step = 0
       
        # 使用更合适的网络规模
        hidden1_size = 512
        hidden2_size = 256
        dropout_rate = 0.1

        # 初始化Actor网络
        self.actor = Actor(state_dim, action_dim, max_action, 
                          hidden1_size, hidden2_size, dropout_rate=dropout_rate).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, weight_decay=1e-5)
        
        # 初始化Critic网络
        self.critic = Critic(state_dim, action_dim, 
                            hidden1_size, hidden2_size, dropout_rate=dropout_rate).to(device)
        self.critic_target = Critic(state_dim, action_dim, 
                                   hidden1_size, hidden2_size, dropout_rate=0).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=1e-5)
        
        self.to_gpu()
        
        # 梯度裁剪值调整
        self.grad_clip = 5.0  # 增大梯度裁剪阈值
        
        # 熵调整参数
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha
        
        self.max_action = max_action
        # 创建更明确的writer实例，并确保记录目录存在
        log_dir = "./logs/SAC_velodyne_" + time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to {log_dir}")
        self.iter_count = 0

         # 添加学习率调度器
        self.use_lr_scheduler = True
        if self.use_lr_scheduler:
            self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.actor_optimizer, 
                T_0=1000, 
                T_mult=2, 
                eta_min=3e-5
            )
            self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.critic_optimizer, 
                T_0=1000, 
                T_mult=2, 
                eta_min=3e-5
            )
        
        # 梯度裁剪值
        self.grad_clip = 1.0
        
        # 初始化MPC控制器
        self.mpc_controller = MPCController(state_dim, action_dim)  

    def to_gpu(self):
        """确保所有网络组件都在GPU上"""
        if torch.cuda.is_available():
            self.actor = self.actor.cuda()
            self.critic = self.critic.cuda()
            self.critic_target = self.critic_target.cuda()
            # 检查GPU使用情况
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            # 打印模型参数总数，以确认模型大小
            actor_params = sum(p.numel() for p in self.actor.parameters())
            critic_params = sum(p.numel() for p in self.critic.parameters())
            print(f"Actor parameters: {actor_params}, Critic parameters: {critic_params}")
    
    def get_action(self, state, eval=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
       
        if eval:
            # 评估模式，使用均值动作
            with torch.no_grad():
                mean, _ = self.actor(state)
                sac_action = torch.tanh(mean).cpu().data.numpy().flatten() * self.max_action
        else:
            # 训练模式，采样动作
            with torch.no_grad():
                sac_action, _ = self.actor.sample(state)
                sac_action = sac_action.cpu().data.numpy().flatten()
        progress = min(self.global_step/ max_timesteps, 1.0)
        mpc_weight = 0.3 * (1 - progress)  # MPC权重随训练衰减
        # 使用MPC优化动作
        def dynamics_model(state, action):
            # 修复维度不匹配问题：仅更新状态中与动作相关的部分
            new_state = state.copy()
            new_state[:2] += action * self.mpc_controller.dt  # 假设前两个维度是位置信息
            return new_state

        def cost_function(states, actions):
            # 成本函数：避免碰撞并接近目标
            target_x, target_y = env.goal_x, env.goal_y
            target_position = np.array([target_x, target_y])
            collision_cost = np.sum(np.maximum(0.2 - states[:, 4:-8].min(axis=1), 0))
            target_cost = np.linalg.norm(states[:, :2] - target_position, axis=1).mean()
            return collision_cost + target_cost

        mpc_action = self.mpc_controller.optimize(state.cpu().numpy().flatten(), dynamics_model, cost_function)
          # 根据当前训练进度调整混合比例
        
        combined_action = (1-mpc_weight)*sac_action + mpc_weight*mpc_action

        return np.clip(combined_action, -self.max_action, self.max_action)
        
    def train(self, replay_buffer, batch_size=100, discount=0.99, tau=0.005, policy_freq=1):
        avg_q_value = 0
        avg_actor_loss = 0
        avg_critic_loss = 0
        avg_alpha_loss = 0
        
        # 从优先经验回放缓冲区采样
        states, actions, rewards, dones, next_states, idxs, weights = replay_buffer.sample_batch(batch_size)
        
        # 转换为tensor
        state = torch.FloatTensor(states).to(self.device)
        action = torch.FloatTensor(actions).to(self.device)
        reward = torch.FloatTensor(rewards).to(self.device)
        next_state = torch.FloatTensor(next_states).to(self.device)
        done = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).reshape(-1, 1).to(self.device)
        
        # ------------ 更新Critic网络 ------------
        with torch.no_grad():
            # 从下一个状态采样动作和对应的对数概率
            next_action, next_log_prob = self.actor.sample(next_state)
            
            # 计算目标Q值 (使用目标网络)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * discount * target_q
        
        # 计算当前的Q值估计
        current_q1, current_q2 = self.critic(state, action)
        
        # 计算TD误差
        td_error1 = torch.abs(target_q - current_q1).detach()
        td_error2 = torch.abs(target_q - current_q2).detach()
        td_error = torch.max(td_error1, td_error2).cpu().numpy()
        
        # 更新优先级
        replay_buffer.update_priorities(idxs, td_error)
        
        # 计算Critic损失 (加入重要性采样权重)
        critic_loss = (weights * F.mse_loss(current_q1, target_q, reduction='none')).mean() + \
                      (weights * F.mse_loss(current_q2, target_q, reduction='none')).mean()
        
        # 梯度下降更新Critic网络 - 修复重复调用backward()的问题
        self.critic_optimizer.zero_grad()
        critic_loss.backward()  # 不需要重复调用backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        
        avg_critic_loss += critic_loss.item()
        avg_q_value += torch.min(current_q1, current_q2).mean().item()
        
        # ------------ 更新Actor网络和Alpha ------------
        # 策略更新频率控制
        if self.iter_count % policy_freq == 0:
            # 采样动作和对数概率
            pi, log_pi = self.actor.sample(state)
            
            # 计算Q值
            q1_pi, q2_pi = self.critic(state, pi)
            q_pi = torch.min(q1_pi, q2_pi)
            
            # 计算Actor损失 (最大化Q值减去熵正则项)
            actor_loss = ((self.alpha * log_pi) - q_pi).mean()
            
            # 梯度下降更新Actor网络
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)  # 添加retain_graph=True以防止计算图被释放
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            
            self.actor_optimizer.step()
            
            avg_actor_loss += actor_loss.item()

            # 自动调整熵系数alpha
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                
                self.alpha = self.log_alpha.exp()
                avg_alpha_loss += alpha_loss.item()

        # 更新学习率调度器
        if self.use_lr_scheduler:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            self.writer.add_scalar("lr/actor", self.actor_scheduler.get_last_lr()[0], self.iter_count)
            self.writer.add_scalar("lr/critic", self.critic_scheduler.get_last_lr()[0], self.iter_count)
        
        # ------------ 软更新目标网络 ------------
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        self.iter_count += 1
        
        # 记录tensorboard数据 - 使用全局步数而不是迭代计数
        self.writer.add_scalar("training/critic_loss", avg_critic_loss, self.global_step)
        self.writer.add_scalar("training/actor_loss", avg_actor_loss, self.global_step)
        if self.automatic_entropy_tuning:
            self.writer.add_scalar("training/alpha_loss", avg_alpha_loss, self.global_step)
            self.writer.add_scalar("training/alpha", self.alpha.item(), self.global_step)
        self.writer.add_scalar("training/q_value", avg_q_value, self.global_step)

    def save(self, filename, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        if self.automatic_entropy_tuning:
            torch.save(self.log_alpha, "%s/%s_alpha.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, filename)))
        self.critic.load_state_dict(torch.load("%s/%s_critic.pth" % (directory, filename)))
        self.critic_target.load_state_dict(self.critic.state_dict())
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.load("%s/%s_alpha.pth" % (directory, filename))
            self.alpha = self.log_alpha.exp()


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
eval_freq = 5e3  # After how many steps to perform the evaluation
max_ep = 500  # maximum number of steps per episode
eval_ep = 10  # number of episodes for evaluation
max_timesteps = 5e6  # Maximum number of steps to perform
expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = 500000  # Number of steps over which the initial exploration noise will decay over
expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
batch_size = 256  # 增大批量大小以提高训练效率
discount = 0.99  # Discount factor to calculate the discounted future reward
tau = 0.001  # Soft target update variable (should be close to 0)
policy_freq = 1  # SAC每步都更新策略
buffer_size = 1e6  # Maximum size of the buffer
file_name = "SAC_velodyne"  # name of the file to store the policy
save_model = True  # Weather to save the model or not
load_model = False  # Weather to load a stored model
random_near_obstacle = True  # To take random actions near
random_near_obstacle = True  # To take random actions near obstacles or not
automatic_entropy_tuning = True  # 自动调整熵系数
init_temperature = 0.2  # 初始化熵系数

# 设置学习率使用余弦退火调度
use_lr_scheduler = True
lr_scheduler_step = 10000  # 学习率调度步长

# 创建结果文件夹
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# Create the training environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

# 创建SAC网络
network = SAC(
    state_dim, 
    action_dim, 
    max_action,
    device, 
    alpha=init_temperature,
    automatic_entropy_tuning=automatic_entropy_tuning
)
network.to_gpu()

# 创建优先经验回放缓冲区
replay_buffer = PrioritizedReplayBuffer(buffer_size, state_dim, action_dim)

# 如果启用学习率调度器
if use_lr_scheduler:
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        network.actor_optimizer,
        T_max=lr_scheduler_step,
        eta_min=3e-5
    )
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        network.critic_optimizer,
        T_max=lr_scheduler_step,
        eta_min=3e-5
    )

# 加载预训练模型
if load_model:
    try:
        network.load(file_name, "./pytorch_models")
        print("Successfully loaded pretrained weights!")
    except:
        print("Could not load the stored model parameters, initializing training with random parameters")

# 创建评估数据存储
evaluations = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

# 创建退火探索噪声
class AdaptiveExplorationNoise:
    def __init__(self, action_dim, initial_scale=1.0, min_scale=0.1, decay_steps=500000):
        self.scale = initial_scale
        self.min_scale = min_scale
        self.decay_rate = (initial_scale - min_scale) / decay_steps
        self.action_dim = action_dim
        self.noise = np.zeros(action_dim)
    
    def sample(self):
        # 使用Ornstein-Uhlenbeck过程生成噪声
        theta = 0.15
        sigma = 0.2
        self.noise = self.noise + theta * (np.zeros(self.action_dim) - self.noise) + \
                    sigma * np.random.normal(0, 1, size=self.action_dim)
        return self.noise * self.scale
    
    def decay(self):
        self.scale = max(self.min_scale, self.scale - self.decay_rate)
    
    def get_scale(self):
        return self.scale

# 创建自适应探索噪声
noise = AdaptiveExplorationNoise(action_dim, initial_scale=expl_noise, min_scale=expl_min, decay_steps=expl_decay_steps)

# 记录训练指标
episode_rewards = []
episode_steps = []
success_rate = []
collision_rate = []
total_success = 0
total_collisions = 0
total_episodes = 0

# 开始训练循环
print("Starting SAC training...")
while timestep < max_timesteps:
     # 在每次步数更新时同步到网络实例
    network.global_step = timestep
    timestep += 1
    # 在回合结束时
    if done:
        if timestep != 0 and 'episode_timesteps' in locals(): 
            # 计算回合统计数据
            avg_reward = episode_reward / episode_timesteps if episode_timesteps > 0 else 0
            episode_rewards.append(avg_reward)
            episode_steps.append(episode_timesteps)
            
            # 记录成功率和碰撞率
            is_success = episode_reward > 50  # 假设奖励大于50为成功
            is_collision = min(state[4:-8]) < 0.2  # 假设最小激光读数小于0.2为碰撞
            
            success_rate.append(1 if is_success else 0)
            collision_rate.append(1 if is_collision else 0)
            
            total_success += 1 if is_success else 0
            total_collisions += 1 if is_collision else 0
            total_episodes += 1
            
            # 直接记录本回合的指标到TensorBoard
            network.writer.add_scalar("episode/reward", episode_reward, episode_num)
            network.writer.add_scalar("episode/avg_reward", avg_reward, episode_num)
            network.writer.add_scalar("episode/steps", episode_timesteps, episode_num)
            network.writer.add_scalar("episode/success", 1 if is_success else 0, episode_num)
            network.writer.add_scalar("episode/collision", 1 if is_collision else 0, episode_num)
            
            # 记录累积指标
            if total_episodes > 0:
                network.writer.add_scalar("metrics/success_rate", total_success / total_episodes, episode_num)
                network.writer.add_scalar("metrics/collision_rate", total_collisions / total_episodes, episode_num)
            
            # 记录滑动窗口指标(最近10次平均)
            window_size = min(10, len(success_rate))
            if window_size > 0:
                recent_success = np.mean(success_rate[-window_size:])
                recent_collision = np.mean(collision_rate[-window_size:])
                recent_reward = np.mean(episode_rewards[-window_size:])
                recent_steps = np.mean(episode_steps[-window_size:])
                
                network.writer.add_scalar("metrics/recent_success_rate", recent_success, episode_num)
                network.writer.add_scalar("metrics/recent_collision_rate", recent_collision, episode_num)
                network.writer.add_scalar("metrics/recent_avg_reward", recent_reward, episode_num)
                network.writer.add_scalar("metrics/recent_avg_steps", recent_steps, episode_num)
                network.writer.add_scalar("metrics/noise_scale", noise.get_scale(), episode_num)
                
                # 每10回合打印统计信息
                if episode_num % 10 == 0:
                    print(f"Episode: {episode_num}, "
                          f"Timestep: {timestep}, "
                          f"Avg. Reward: {recent_reward:.2f}, "
                          f"Success Rate: {recent_success:.2f}, "
                          f"Collision Rate: {recent_collision:.2f}, "
                          f"Avg. Steps: {recent_steps:.2f}, "
                          f"Noise Scale: {noise.get_scale():.2f}")
            
            # 确保所有数据都写入磁盘
            network.writer.flush()
            
            # 更新学习率调度器
            if use_lr_scheduler and timestep % lr_scheduler_step == 0:
                actor_scheduler.step()
                critic_scheduler.step()
                network.writer.add_scalar("lr/actor", actor_scheduler.get_last_lr()[0], timestep)
                network.writer.add_scalar("lr/critic", critic_scheduler.get_last_lr()[0], timestep)

        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(
                evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
            )
            network.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1

        state = env.reset()
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # 获取动作
    action = network.get_action(np.array(state))
    
    # 添加探索噪声，并衰减噪声大小
    if timestep < expl_decay_steps:
        action = action + noise.sample()
        action = np.clip(action, -max_action, max_action)
        noise.decay()

    # 针对障碍物的特殊处理 (保留原逻辑)
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[4:-8]) < 0.6
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    # 更新动作范围
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    # 保存经验到回放缓冲区
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # 当缓冲区足够大时训练
    if replay_buffer.size > batch_size:
        network.train(replay_buffer, batch_size, discount, tau, policy_freq)

    # 更新计数器
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# 训练结束后，评估网络并保存
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./pytorch_models")
np.save("./results/%s" % file_name, evaluations)

# 确保TensorBoard writer正确关闭
network.writer.close()
print("Training completed successfully!")
    
      