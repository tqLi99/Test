import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv

def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.select_action(np.array(state), deterministic=True)
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

class SACNetworks(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=800):
        super(SACNetworks, self).__init__()
        
        # Policy Network (Actor)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 600),
            nn.ReLU()
        )
        self.mean = nn.Linear(600, action_dim)
        self.log_std = nn.Linear(600, action_dim)
        
        # Q1 Network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 600),
            nn.ReLU(),
            nn.Linear(600, 1)
        )
        
        # Q2 Network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 600),
            nn.ReLU(),
            nn.Linear(600, 1)
        )
        
    def pi(self, state, deterministic=False):
        policy_output = self.policy(state)
        mean = self.mean(policy_output)
        log_std = self.log_std(policy_output)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
        
        action = torch.tanh(action)
        log_prob = dist.log_prob(action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)
    
    def q2_forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q2(sa)

class ArtificialPotentialField:
    def __init__(self, sensor_range=5.0, eta_att=1.0, eta_rep=1.0, d_safe=0.5):
        self.sensor_range = sensor_range
        self.eta_att = eta_att
        self.eta_rep = eta_rep
        self.d_safe = d_safe
    
    def compute_force(self, laser_data, target_direction):
        # 计算引力（朝向目标方向）
        attractive_force = self.eta_att * target_direction
        
        # 计算斥力（来自障碍物）
        repulsive_force = np.zeros(2)
        for i, distance in enumerate(laser_data):
            if distance < self.sensor_range:
                angle = i * 2 * np.pi / len(laser_data)
                magnitude = self.eta_rep * (1/distance - 1/self.sensor_range) * 1/distance**2
                repulsive_force[0] += magnitude * np.cos(angle)
                repulsive_force[1] += magnitude * np.sin(angle)
        
        # 合成力
        total_force = attractive_force - repulsive_force
        return total_force

class SAC:
    def __init__(self, state_dim, action_dim, max_action,
                 hidden_dim=800,
                 lr=3e-4,
                 alpha=0.2,
                 gamma=0.99,
                 tau=0.005,
                 auto_entropy_tuning=True):
        
        self.networks = SACNetworks(state_dim, action_dim, hidden_dim).to(device)
        self.networks_target = SACNetworks(state_dim, action_dim, hidden_dim).to(device)
        self.networks_target.load_state_dict(self.networks.state_dict())
        
        self.policy_optimizer = torch.optim.Adam(
            list(self.networks.policy.parameters()) +
            list(self.networks.mean.parameters()) +
            list(self.networks.log_std.parameters()),
            lr=lr
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.networks.q1.parameters()) +
            list(self.networks.q2.parameters()),
            lr=lr
        )
        
        self.max_action = max_action
        self.apf = ArtificialPotentialField()
        self.writer = SummaryWriter()
        self.iter_count = 0
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        
        # 自动调整熵值
        if auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.auto_entropy_tuning = auto_entropy_tuning
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action, _ = self.networks.pi(state, deterministic)
            
            # 获取激光雷达数据（假设在state的特定位置）
            laser_data = state[0, 4:-8].cpu().numpy()
            # 使用人工势场法修正动作
            target_direction = action.cpu().numpy()[0]
            apf_force = self.apf.compute_force(laser_data, target_direction)
            
            # 合并SAC动作和APF力
            modified_action = action.cpu().numpy()[0] + 0.3 * apf_force
            return np.clip(modified_action, -self.max_action, self.max_action)
    
    def train(self, replay_buffer, iterations, batch_size=100):
        av_Q = 0
        av_loss = 0
        
        for _ in range(iterations):
            # Sample from replay buffer
            state, action, reward, done, next_state = replay_buffer.sample_batch(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(done).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            
            # 计算目标Q值
            with torch.no_grad():
                next_action, next_log_pi = self.networks.pi(next_state)
                target_q1 = self.networks_target.q1_forward(next_state, next_action)
                target_q2 = self.networks_target.q2_forward(next_state, next_action)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
                target_q = reward + (1 - done) * self.gamma * target_q
            
            # 更新Q网络
            current_q1 = self.networks.q1_forward(state, action)
            current_q2 = self.networks.q2_forward(state, action)
            q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
            
            # 更新策略网络
            pi, log_pi = self.networks.pi(state)
            q1_pi = self.networks.q1_forward(state, pi)
            q2_pi = self.networks.q2_forward(state, pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            policy_loss = (self.alpha * log_pi - min_q_pi).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # 更新自动熵值
            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()
            
            # 软更新目标网络
            for param, target_param in zip(self.networks.parameters(), self.networks_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            av_Q += min_q_pi.mean().item()
            av_loss += q_loss.item()
        
        self.iter_count += 1
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
    
    def save(self, filename, directory):
        torch.save(self.networks.state_dict(), "%s/%s_sac.pth" % (directory, filename))
    
    def load(self, filename, directory):
        self.networks.load_state_dict(
            torch.load("%s/%s_sac.pth" % (directory, filename))
        )
        self.networks_target.load_state_dict(self.networks.state_dict())

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
eval_freq = 5e3
max_ep = 500
eval_ep = 10
max_timesteps = 5e6
batch_size = 40
gamma = 0.99999
tau = 0.005
buffer_size = 1e6
file_name = "SAC_velodyne"
save_model = True
load_model = False

# 创建必要的目录
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# 创建环境
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)

# 设置随机种子
torch.manual_seed(seed)
np.random.seed(seed)

# 创建SAC网络和经验回放缓冲区
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

network = SAC(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(buffer_size, seed)

if load_model:
    try:
        network.load(file_name, "./pytorch_models")
    except:
        print("Could not load the stored model parameters, initializing training with random parameters")

# 创建评估数据存储
evaluations = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

# 训练循环
while timestep < max_timesteps:
    if done:
        if timestep != 0:
            network.train(replay_buffer, episode_timesteps, batch_size)
            
        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
            network.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1
        
        state = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
    
    # 选择动作
    action = network.select_action(np.array(state))
    
    # 执行动作
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    
    episode_reward += reward
    
    # 存储转换到回放缓冲区
    replay_buffer.add(state, action, reward, done_bool, next_state)
    
    # 更新计数器和状态
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# 训练结束后进行最终评估并保存模型
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./pytorch_models")
np.save("./results/%s" % file_name, evaluations)