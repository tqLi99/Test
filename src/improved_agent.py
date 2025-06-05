"""
数值稳定性改进的MASAC智能体
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class NumericallyStableGaussianPolicy(nn.Module):
    """数值稳定的高斯策略网络"""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256,
                 log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim

        # 使用更稳定的激活函数和归一化
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # 改进的权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """改进的权重初始化"""
        if isinstance(m, nn.Linear):
            # 使用Xavier初始化以保持梯度稳定
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        try:
            # 输入标准化
            state = torch.clamp(state, -10, 10)  # 防止极值

            x = self.backbone(state)
            mean = self.mean_head(x)
            log_std = self.log_std_head(x)

            # 确保数值稳定性
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

            return mean, log_std

        except Exception as e:
            logger.error(f"策略网络前向传播失败: {e}")
            # 返回安全的默认值
            batch_size = state.shape[0]
            device = state.device
            return (torch.zeros(batch_size, self.action_dim, device=device),
                    torch.zeros(batch_size, self.action_dim, device=device))

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作"""
        try:
            mean, log_std = self.forward(state)
            std = log_std.exp()

            # 检查数值稳定性
            if torch.isnan(mean).any() or torch.isnan(std).any():
                logger.warning("检测到NaN值，使用默认值")
                mean = torch.zeros_like(mean)
                std = torch.ones_like(std)

            # 限制标准差的范围
            std = torch.clamp(std, min=1e-6, max=1.0)

            normal = Normal(mean, std)

            # 重参数化技巧
            x_t = normal.rsample()
            action = torch.tanh(x_t)

            # 计算对数概率
            log_prob = normal.log_prob(x_t)

            # 校正对数概率 (Jacobian adjustment)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)

            # 检查最终结果
            if torch.isnan(log_prob).any():
                logger.warning("对数概率出现NaN，使用默认值")
                log_prob = torch.zeros_like(log_prob)

            return action, log_prob, torch.tanh(mean)

        except Exception as e:
            logger.error(f"动作采样失败: {e}")
            # 返回安全的默认值
            batch_size = state.shape[0]
            device = state.device
            return (torch.zeros(batch_size, self.action_dim, device=device),
                    torch.zeros(batch_size, 1, device=device),
                    torch.zeros(batch_size, self.action_dim, device=device))


class NumericallyStableQNetwork(nn.Module):
    """数值稳定的Q网络"""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # 第一个Q网络
        self.q1_net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        # 第二个Q网络
        self.q2_net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        try:
            # 输入处理和数值稳定性检查
            state = torch.clamp(state, -10, 10)
            action = torch.clamp(action, -1, 1)

            batch_size = state.size(0)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            if action.size(0) != batch_size:
                action = action.expand(batch_size, -1)

            xu = torch.cat([state, action], 1)

            # 计算两个Q值
            q1 = self.q1_net(xu)
            q2 = self.q2_net(xu)

            # 检查数值稳定性
            if torch.isnan(q1).any() or torch.isnan(q2).any():
                logger.warning("Q网络输出包含NaN，使用默认值")
                q1 = torch.zeros_like(q1)
                q2 = torch.zeros_like(q2)

            return q1, q2

        except Exception as e:
            logger.error(f"Q网络前向传播失败: {e}")
            batch_size = state.shape[0]
            device = state.device
            return (torch.zeros(batch_size, 1, device=device),
                    torch.zeros(batch_size, 1, device=device))


class ImprovedMASACAgent:
    """改进的MASAC智能体"""

    def __init__(self, input_dim: int, action_dim: int, config, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 从配置中获取超参数
        self.gamma = config.training.gamma
        self.tau = config.training.tau
        self.alpha = config.training.alpha
        self.auto_entropy_tuning = config.training.auto_entropy_tuning
        self.max_grad_norm = config.training.max_grad_norm

        # 创建网络
        self.policy = NumericallyStableGaussianPolicy(
            input_dim, action_dim, config.training.hidden_dim
        ).to(self.device)

        self.critic = NumericallyStableQNetwork(
            input_dim, action_dim, config.training.hidden_dim
        ).to(self.device)

        self.critic_target = NumericallyStableQNetwork(
            input_dim, action_dim, config.training.hidden_dim
        ).to(self.device)

        # 优化器
        self.policy_optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config.training.learning_rate,
            weight_decay=1e-4
        )

        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(),
            lr=config.training.learning_rate,
            weight_decay=1e-4
        )

        # 硬拷贝参数到目标网络
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # 熵权重处理
        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=config.training.learning_rate)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(self.alpha, device=self.device, requires_grad=False)

        # 混合精度训练
        self.use_amp = torch.cuda.is_available() and hasattr(torch, 'amp')
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        # 性能监控
        self.update_count = 0
        self.loss_history = {'critic': [], 'policy': [], 'alpha': []}

        logger.info(f"MASAC智能体初始化完成，设备: {self.device}")

    def act(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """根据状态选择动作"""
        try:
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

            with torch.no_grad():
                if evaluate:
                    # 评估模式使用确定性动作
                    _, _, action = self.policy.sample(state)
                    return action.detach().cpu().numpy()[0]
                else:
                    # 训练模式使用随机动作
                    action, _, _ = self.policy.sample(state)
                    return action.detach().cpu().numpy()[0]

        except Exception as e:
            logger.error(f"动作选择失败: {e}")
            return np.zeros(self.policy.action_dim)

    def update(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray,
               next_state: np.ndarray, done: np.ndarray,
               weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float]:
        """更新网络参数"""
        try:
            self.update_count += 1

            # 数据预处理
            state, action, reward, next_state, done, weights = self._preprocess_batch(
                state, action, reward, next_state, done, weights
            )

            # 更新Critic网络
            td_errors, critic_loss = self._update_critic(
                state, action, reward, next_state, done, weights
            )

            # 根据延迟策略更新Policy网络
            policy_loss = 0.0
            alpha_loss = 0.0

            if self.update_count % 2 == 0:  # 延迟策略更新
                policy_loss = self._update_policy(state.detach().clone())

                if self.auto_entropy_tuning:
                    alpha_loss = self._update_alpha(state.detach().clone())

                # 软更新目标网络
                self._soft_update()

            # 记录损失
            self.loss_history['critic'].append(critic_loss)
            self.loss_history['policy'].append(policy_loss)
            self.loss_history['alpha'].append(alpha_loss)

            return td_errors.detach().cpu().numpy(), critic_loss, policy_loss

        except Exception as e:
            logger.error(f"网络更新失败: {e}")
            return np.zeros(state.shape[0]), 0.0, 0.0

    def _preprocess_batch(self, state, action, reward, next_state, done, weights):
        """预处理批次数据"""
        # 转换为张量
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)

        # 处理奖励形状
        reward_np = np.array(reward)
        if reward_np.ndim == 1:
            reward_np = reward_np.reshape(-1, 1)
        reward = torch.FloatTensor(reward_np).to(self.device)

        next_state = torch.FloatTensor(next_state).to(self.device)

        # 处理done标志
        done_np = np.array(done)
        if done_np.ndim == 1:
            done_np = done_np.reshape(-1, 1)
        done = torch.FloatTensor(done_np).to(self.device)

        # 重要性采样权重
        if weights is not None:
            weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        else:
            weights = torch.ones(state.size(0), 1).to(self.device)

        return state, action, reward, next_state, done, weights

    def _update_critic(self, state, action, reward, next_state, done, weights):
        """更新Critic网络"""
        try:
            # 计算目标Q值
            with torch.no_grad():
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

            # TD误差
            td_errors = torch.abs(current_q1 - q_target).squeeze()

            # 更新网络
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

        except Exception as e:
            logger.error(f"Critic更新失败: {e}")
            return torch.zeros(state.shape[0]), 0.0

    def _update_policy(self, state):
        """更新Policy网络"""
        try:
            pi, log_pi, _ = self.policy.sample(state)
            qf1_pi, qf2_pi = self.critic(state, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

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

        except Exception as e:
            logger.error(f"Policy更新失败: {e}")
            return 0.0

    def _update_alpha(self, state):
        """更新熵权重"""
        try:
            with torch.no_grad():
                _, log_pi, _ = self.policy.sample(state)

            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

            return alpha_loss.detach().item()

        except Exception as e:
            logger.error(f"Alpha更新失败: {e}")
            return 0.0

    def _soft_update(self):
        """软更新目标网络"""
        try:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        except Exception as e:
            logger.error(f"软更新失败: {e}")

    def save_model(self, path: str):
        """保存模型"""
        try:
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'update_count': self.update_count,
                'loss_history': self.loss_history
            }, path)
            logger.info(f"模型已保存到: {path}")
        except Exception as e:
            logger.error(f"模型保存失败: {e}")

    def load_model(self, path: str):
        """加载模型"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.update_count = checkpoint.get('update_count', 0)
            self.loss_history = checkpoint.get('loss_history', {'critic': [], 'policy': [], 'alpha': []})
            logger.info(f"模型已从 {path} 加载")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")