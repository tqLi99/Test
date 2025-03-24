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

# 简化的优先经验回放缓冲区
class PrioritizedReplayBuffer:
    def __init__(self, max_size, input_shape, action_shape, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.alpha = alpha  # 增大alpha值，更偏向高TD误差的样本
        self.beta = beta    # 降低初始beta值，逐步增加重要性采样权重
        self.beta_increment = beta_increment
        self.epsilon = 1e-5  # 避免优先级为0

        self.states = np.zeros((self.max_size, input_shape))
        self.actions = np.zeros((self.max_size, action_shape))
        self.rewards = np.zeros((self.max_size, 1))
        self.next_states = np.zeros((self.max_size, input_shape))
        self.dones = np.zeros((self.max_size, 1))
        self.priorities = np.ones((self.max_size, 1))  # 初始化为1而非0

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
            # 使用segment-based优先采样，提高稳定性
            priorities = self.priorities[:self.size]
            probabilities = priorities ** self.alpha
            probabilities = probabilities / probabilities.sum()

            # 分段采样以减少方差
            segment_size = self.size // batch_size
            idxs = []
            for i in range(batch_size):
                segment_start = i * segment_size
                segment_end = (i + 1) * segment_size
                if segment_end > self.size:
                    segment_end = self.size
                
                # 在每个段内基于优先级采样
                segment_probs = probabilities[segment_start:segment_end] / probabilities[segment_start:segment_end].sum()
                idx = np.random.choice(np.arange(segment_start, segment_end), p=segment_probs.flatten()) if segment_end > segment_start else segment_start
                idxs.append(idx)
            
            idxs = np.array(idxs)

        states = self.states[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_states = self.next_states[idxs]
        dones = self.dones[idxs]

        # 计算重要性权重
        weights = (self.size * probabilities[idxs]) ** (-self.beta)
        weights = weights / weights.max()
        weights = np.array(weights, dtype=np.float32)

        # 增加beta以逐渐减少重要性采样偏差
        self.beta = min(1.0, self.beta + self.beta_increment)

        return states, actions, rewards, dones, next_states, idxs, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.abs(priorities) + self.epsilon  # 添加绝对值确保正值
        for idx, priority in zip(idxs, priorities):
            self.priorities[idx] = priority

def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col_count = 0
    success_count = 0
    
    for ep in range(eval_episodes):
        step_count = 0
        state = env.reset()
        done = False
        episode_reward = 0
        had_collision = False  # 每个评估回合的碰撞标志
        had_success = False    # 每个评估回合的成功标志
        
        while not done and step_count < 501:
            action = network.get_action(np.array(state), eval=True)
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, target = env.step(a_in)
            episode_reward += reward
            step_count += 1
            
            # 检测碰撞 - 保存到 had_collision 变量
            if reward < -90:  
                had_collision = True
                
            # 检测成功 - 保存到 had_success 变量
            if target:
                had_success = True
        
        # 回合结束后统计
        avg_reward += episode_reward
        
        # 如果回合中发生了碰撞，增加碰撞计数
        if had_collision:
            col_count += 1
            
        # 如果回合中达成了目标，增加成功计数
        if had_success:
            success_count += 1
        
    # 计算平均值
    avg_reward /= eval_episodes
    collision_rate = col_count / eval_episodes
    success_rate = success_count / eval_episodes
    
    print("..............................................")
    print(
        "Evaluation - Epoch %i: Avg Reward: %.2f, Collision Rate: %.2f, Success Rate: %.2f"
        % (epoch, avg_reward, collision_rate, success_rate)
    )
    print("..............................................")
    
    # 记录评估指标到TensorBoard
    network.writer.add_scalar("evaluation/avg_reward", avg_reward, epoch)
    network.writer.add_scalar("evaluation/collision_rate", collision_rate, epoch)
    network.writer.add_scalar("evaluation/success_rate", success_rate, epoch)
    
    return avg_reward, collision_rate, success_rate
# 简化的Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 使用更简单的网络结构，减少过拟合风险
        self.layer_norm = nn.LayerNorm(state_dim)
        
        # 使用简单的全连接网络
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 均值和标准差输出层
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # 使用合适的初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier初始化，适合tanh激活函数
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, state):
        # 输入规范化
        x = self.layer_norm(state)
        
        # 前向传播
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 均值和标准差计算
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        
        # 创建正态分布
        normal = Normal(mean, std)
        
        # 重参数化采样
        x = normal.rsample()
        
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

# 简化的Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # 输入层规范化
        self.layer_norm = nn.LayerNorm(state_dim)
        
        # Q1网络
        self.q1_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2网络 (双Q学习)
        self.q2_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state, action):
        # 规范化状态
        state = self.layer_norm(state)
        
        # 合并状态和动作
        x = torch.cat([state, action], dim=1)
        
        # 前向传播获取Q值
        q1 = self.q1_network(x)
        q2 = self.q2_network(x)
        
        return q1, q2

# 改进的SAC类
class SAC(object):
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        max_action, 
        device, 
        alpha=0.2,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        automatic_entropy_tuning=True,
        hidden_dim=256,
        target_update_interval=1
    ):
        self.device = device
        self.gamma = gamma  # 折扣因子
        self.tau = tau      # 目标网络更新率
        self.alpha = alpha  # 熵系数
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        self.global_step = 0
        self.update_count = 0
       
        # 初始化Actor网络
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # 初始化Critic网络
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        # 硬拷贝确保初始权重相同
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # 关闭梯度更新
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # 熵调整参数
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha
        
        self.max_action = max_action
        
        # 创建TensorBoard记录器
        log_dir = "./logs/ImprovedSAC_" + time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to {log_dir}")
        
        # 梯度裁剪值
        self.grad_clip = 1.0
    
    def get_action(self, state, eval=False):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            
            if eval:
                # 评估模式，使用确定性动作
                mean, _ = self.actor(state)
                action = torch.tanh(mean) * self.max_action
                return action.cpu().numpy().flatten()
            else:
                # 训练模式，带探索的随机采样
                action, _ = self.actor.sample(state)
                return action.cpu().numpy().flatten()
    
    def train(self, replay_buffer, batch_size=256):
        # 从优先经验回放缓冲区采样
        states, actions, rewards, dones, next_states, idxs, weights = replay_buffer.sample_batch(batch_size)
        
        # 转换为tensor
        state = torch.FloatTensor(states).to(self.device)
        action = torch.FloatTensor(actions).to(self.device)
        reward = torch.FloatTensor(rewards).to(self.device)
        next_state = torch.FloatTensor(next_states).to(self.device)
        done = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).reshape(-1, 1).to(self.device)
        
        with torch.no_grad():
            # 从下一个状态采样动作和对应的对数概率
            next_action, next_log_prob = self.actor.sample(next_state)
            
            # 计算目标Q值
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # 计算当前的Q值估计
        current_q1, current_q2 = self.critic(state, action)
        
        # 计算TD误差用于更新优先级
        td_error1 = torch.abs(target_q - current_q1).detach()
        td_error2 = torch.abs(target_q - current_q2).detach()
        td_error = torch.max(td_error1, td_error2).cpu().numpy()
        
        # 更新优先级
        replay_buffer.update_priorities(idxs, td_error)
        
        # 计算Critic损失 (考虑重要性采样权重)
        critic_loss1 = (weights * F.mse_loss(current_q1, target_q, reduction='none')).mean()
        critic_loss2 = (weights * F.mse_loss(current_q2, target_q, reduction='none')).mean()
        critic_loss = critic_loss1 + critic_loss2
        
        # 更新Critic网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        
        # 延迟更新Actor网络
        actor_loss = 0
        if self.update_count % self.target_update_interval == 0:
            # 采样动作和对数概率
            pi, log_pi = self.actor.sample(state)
            
            # 计算Q值
            q1_pi, q2_pi = self.critic(state, pi)
            q_pi = torch.min(q1_pi, q2_pi)
            
            # 计算Actor损失 (最大化Q值减去熵正则项)
            actor_loss = ((self.alpha * log_pi) - q_pi).mean()
            
            # 更新Actor网络
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()
            
            # 自动调整熵系数alpha
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                
                self.alpha = self.log_alpha.exp()
                
                # 记录alpha相关数据
                self.writer.add_scalar("training/alpha", self.alpha.item(), self.global_step)
                self.writer.add_scalar("training/alpha_loss", alpha_loss.item(), self.global_step)
                
            # 软更新目标网络
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.update_count += 1
        
        # 记录训练数据
        self.writer.add_scalar("training/critic_loss", critic_loss.item(), self.global_step)
        self.writer.add_scalar("training/actor_loss", actor_loss if isinstance(actor_loss, int) else actor_loss.item(), self.global_step)
        self.writer.add_scalar("training/q_value", torch.min(current_q1, current_q2).mean().item(), self.global_step)
        
        # 返回损失值，可用于监控
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss if isinstance(actor_loss, int) else actor_loss.item(),
            "q_value": torch.min(current_q1, current_q2).mean().item()
        }

    def save(self, filename, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        if self.automatic_entropy_tuning:
            torch.save(self.log_alpha, "%s/%s_alpha.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, filename), map_location=self.device))
        self.critic.load_state_dict(torch.load("%s/%s_critic.pth" % (directory, filename), map_location=self.device))
        self.critic_target.load_state_dict(self.critic.state_dict())
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.load("%s/%s_alpha.pth" % (directory, filename), map_location=self.device)
            self.alpha = self.log_alpha.exp()


# 改进的探索噪声类
class AdaptiveExplorationNoise:
    def __init__(self, action_dim, initial_scale=1.0, min_scale=0.3, decay_steps=1000000):
        self.scale = initial_scale
        self.min_scale = min_scale
        self.decay_rate = (initial_scale - min_scale) / decay_steps
        self.action_dim = action_dim
        self.noise = np.zeros(action_dim)
        self.step = 0
    
    def sample(self):
        # 使用Ornstein-Uhlenbeck过程生成噪声
        theta = 0.15
        sigma = 0.3  # 增大噪声强度
        self.noise = self.noise + theta * (np.zeros(self.action_dim) - self.noise) + \
                    sigma * np.random.normal(0, 1, size=self.action_dim)
        return self.noise * self.scale
    
    def decay(self):
        self.step += 1
        # 使用余弦衰减策略
        if self.step % 1000 == 0:  # 每1000步更新一次
            # 更缓慢的衰减
            cosine_decay = 0.5 * (1 + np.cos(np.pi * self.step / 1000000))
            self.scale = self.min_scale + (1.0 - self.min_scale) * cosine_decay
    
    def get_scale(self):
        return self.scale


# 主训练函数
def main():
    # 设置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42  # 更改随机种子
    eval_freq = 5000  # 评估频率
    max_ep_len = 500  # 每个回合最大步数
    eval_episodes = 10
    max_timesteps = 1e6  # 最大训练步数
    
    # 增大批量大小
    batch_size = 256
    
    # SAC超参数
    gamma = 0.99  # 折扣因子
    tau = 0.005   # 软更新系数
    lr = 3e-4     # 学习率
    alpha = 0.2   # 初始熵系数
    hidden_dim = 256  # 隐藏层维度
    
    # 经验回放参数
    buffer_size = 1e6
    
    # 探索噪声参数 - 更缓慢的衰减
    expl_noise = 1.0
    expl_min = 0.3
    expl_decay_steps = 1000000
    
    # 保存和加载设置
    file_name = "ImprovedSAC_velodyne"
    save_model = True
    load_model = False
    
    # 自动熵调整
    automatic_entropy_tuning = True
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 创建环境
    global env
    environment_dim = 20
    robot_dim = 4
    env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
    time.sleep(5)
    
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1.0
    
    # 创建改进的SAC网络
    network = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        alpha=alpha,
        gamma=gamma,
        tau=tau,
        lr=lr,
        automatic_entropy_tuning=automatic_entropy_tuning,
        hidden_dim=hidden_dim,
        target_update_interval=1  # 每步更新Actor
    )
    
    # 创建改进的优先经验回放缓冲区
    replay_buffer = PrioritizedReplayBuffer(
        max_size=buffer_size,
        input_shape=state_dim,
        action_shape=action_dim,
        alpha=0.6,  # 更高的alpha值
        beta=0.4    # 起始beta值
    )
    
    # 创建自适应探索噪声
    noise = AdaptiveExplorationNoise(
        action_dim=action_dim,
        initial_scale=expl_noise,
        min_scale=expl_min,
        decay_steps=expl_decay_steps
    )
    
    # 加载预训练模型
    if load_model:
        try:
            network.load(file_name, "./pytorch_models")
            print("Successfully loaded pretrained weights!")
        except:
            print("Could not load the stored model parameters, initializing training with random parameters")
    
    # 创建记录器
    evaluations = []
    episode_rewards = []
    episode_steps = []
    success_rates = []
    collision_rates = []
    
    # 训练循环计数器
    timestep = 0
    episode_num = 0
    epoch = 1
    
    # 训练循环
    print("Starting improved SAC training...")
    while timestep < max_timesteps:
        # 在每次步数更新时同步到网络实例
        network.global_step = timestep
        
        # 重置环境
        state = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        had_collision = False  # 用于跟踪本回合是否有碰撞
        had_success = False    # 用于跟踪本回合是否成功
        
        # 回合循环
        while not done and episode_timesteps < max_ep_len:
            # 获取动作
            action = network.get_action(np.array(state))
            
            # 添加探索噪声
            action = action + noise.sample()
            action = np.clip(action, -max_action, max_action)
            noise.decay()
            
           # 执行动作
            a_in = [(action[0] + 1) / 2, action[1]]
            next_state, reward, done, target = env.step(a_in)  # 注意：env.step返回的是target

             # 检查是否碰撞或成功
            if reward < -90:  # 根据奖励判断是否碰撞
                had_collision = True
            if target:  # 根据返回的target判断是否成功
                had_success = True
            
            # 记录回合数据
            episode_reward += reward
            episode_timesteps += 1
            
            # 标记回合结束条件
            done_bool = float(done) if episode_timesteps < max_ep_len else 0
            
            # 添加经验到回放缓冲区
            replay_buffer.add(state, action, reward, done_bool, next_state)
            
            # 训练SAC
            if replay_buffer.size > batch_size:
                network.train(replay_buffer, batch_size)
            
            # 更新状态
            state = next_state
            timestep += 1
            
            # 定期评估
            if timestep % eval_freq == 0:
                avg_reward, avg_col, success_rate = evaluate(network, epoch, eval_episodes)
                evaluations.append([avg_reward, avg_col, success_rate])
                if save_model:
                    network.save(file_name, directory="./pytorch_models")
                    # 保存所有评估指标
                    np.savez(
                        f"./results/{file_name}_metrics.npz", 
                        evaluations=np.array(evaluations),
                        rewards=np.array(episode_rewards),
                        steps=np.array(episode_steps),
                        success_rates=np.array(success_rates),
                        collision_rates=np.array(collision_rates)
                )
            epoch += 1
        
        # 回合结束，记录数据
        episode_num += 1
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_timesteps)

        # 更新成功率和碰撞率列表
        success_rates.append(1 if had_success else 0)
        collision_rates.append(1 if had_collision else 0)
    
        # 记录到TensorBoard
        network.writer.add_scalar("episode/reward", episode_reward, episode_num)
        network.writer.add_scalar("episode/steps", episode_timesteps, episode_num)
        network.writer.add_scalar("episode/success", 1 if had_success else 0, episode_num)
        network.writer.add_scalar("episode/collision", 1 if had_collision else 0, episode_num)
        
        # 每10个回合打印统计信息
        if episode_num % 10 == 0:
            # 计算近期平均值
            window = min(10, len(episode_rewards))
            recent_rewards = np.mean(episode_rewards[-window:])
            recent_steps = np.mean(episode_steps[-window:])
            recent_success = np.mean(success_rates[-window:]) if success_rates else 0
            recent_collision = np.mean(collision_rates[-window:]) if collision_rates else 0
        
            print(f"Episode: {episode_num}, Timestep: {timestep}, " 
                  f"Recent {window} Avg Reward: {recent_rewards:.2f}, "
                  f"Success Rate: {recent_success:.2f}, "
                  f"Collision Rate: {recent_collision:.2f}, "
                  f"Avg Steps: {recent_steps:.2f}, "
                  f"Noise Scale: {noise.get_scale():.3f}")
            
             # 记录到TensorBoard
            network.writer.add_scalar("metrics/avg_episode_reward", recent_rewards, episode_num)
            network.writer.add_scalar("metrics/avg_episode_steps", recent_steps, episode_num)
            network.writer.add_scalar("metrics/recent_success_rate", recent_success, episode_num)
            network.writer.add_scalar("metrics/recent_collision_rate", recent_collision, episode_num)
            network.writer.add_scalar("metrics/noise_scale", noise.get_scale(), episode_num)

            # 记录累积指标
            if len(success_rates) > 0:
               total_success_rate = np.mean(success_rates)
               total_collision_rate = np.mean(collision_rates)
               network.writer.add_scalar("metrics/total_success_rate", total_success_rate, episode_num)
               network.writer.add_scalar("metrics/total_collision_rate", total_collision_rate, episode_num)

    # 训练结束，进行最终评估
    final_reward, final_col, final_success = evaluate(network, epoch, eval_episodes=20)
    evaluations.append([final_reward, final_col, final_success])
    
    # 保存模型
    if save_model:
        network.save(file_name, directory="./pytorch_models")
        np.savez(
            f"./results/{file_name}_final_metrics.npz", 
            evaluations=np.array(evaluations),
            rewards=np.array(episode_rewards),
            steps=np.array(episode_steps),
            success_rates=np.array(success_rates),
            collision_rates=np.array(collision_rates)
        )
    
    # 关闭TensorBoard记录器
    network.writer.close()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()