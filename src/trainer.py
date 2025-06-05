"""
改进的训练器 - 修复张量梯度警告版本
"""
import os
import logging
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from config import Config
from improved_environment import ImprovedAdaptiveFormationEnvironment
from improved_agent import ImprovedMASACAgent
from memory_buffer import OptimizedPrioritizedReplayBuffer, SimpleReplayBuffer
from parallel_env import ParallelEnvironmentWrapper

logger = logging.getLogger(__name__)


class ImprovedTrainer:
    """改进的训练器"""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建目录
        os.makedirs('models', exist_ok=True)
        os.makedirs('runs', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

        # 初始化TensorBoard
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(f'runs/improved_masac_{current_time}')

        # 创建环境
        self.env = ImprovedAdaptiveFormationEnvironment(config)

        # 获取观察和动作维度
        test_obs = self.env.reset()
        self.obs_dim = len(test_obs[0])
        self.action_dim = self.env.action_dim
        self.n_agents = self.env.n_agents

        logger.info(f"观察维度: {self.obs_dim}, 动作维度: {self.action_dim}, 智能体数量: {self.n_agents}")

        # 创建智能体
        self.agents = [
            ImprovedMASACAgent(self.obs_dim, self.action_dim, config, self.device)
            for _ in range(self.n_agents)
        ]

        # 创建经验回放缓冲区
        if config.training.use_prioritized_replay:
            self.memory = OptimizedPrioritizedReplayBuffer(
                max_size=config.training.buffer_size
            )
            logger.info("使用优先经验回放")
        else:
            self.memory = SimpleReplayBuffer(max_size=config.training.buffer_size)
            logger.info("使用简单经验回放")

        # 训练统计
        self.episode_rewards = []
        self.success_rate = []
        self.formation_errors = []
        self.collision_counts = []

        # 最佳模型跟踪
        self.best_success_rate = 0.0
        self.best_episode = 0

        # 性能监控
        self.training_start_time = None
        self.episodes_completed = 0

    def train(self) -> Dict[str, Any]:
        """主训练循环"""
        logger.info("开始训练...")
        self.training_start_time = datetime.now()

        try:
            for episode in tqdm(range(1, self.config.training.n_episodes + 1),
                                desc="训练进度"):
                episode_stats = self._train_episode(episode)
                self._update_statistics(episode_stats, episode)

                # 定期评估和保存
                if episode % 50 == 0:
                    self._log_progress(episode)

                if episode % 200 == 0:
                    self._save_models(episode)

                if episode % 500 == 0:
                    self._evaluate_and_visualize(episode)

                # 早停检查
                if self._should_early_stop(episode):
                    logger.info(f"在第 {episode} 回合触发早停")
                    break

            self.episodes_completed = episode

        except KeyboardInterrupt:
            logger.info("训练被用户中断")
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            raise
        finally:
            self._cleanup()

        return self._get_training_summary()

    def _train_episode(self, episode: int) -> Dict[str, Any]:
        """训练单个回合"""
        states = self.env.reset()
        episode_reward = 0.0
        collisions = 0
        steps = 0

        # 每个智能体的损失
        agent_losses = {'critic': [0.0] * self.n_agents,
                        'policy': [0.0] * self.n_agents}
        update_counts = [0] * self.n_agents

        for step in range(self.config.environment.max_steps):
            # 选择动作
            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.act(states[i], evaluate=False)
                actions.append(action)

            # 执行动作
            next_states, rewards, done, info = self.env.step(actions)
            episode_reward += sum(rewards)
            steps = step + 1

            # 记录碰撞
            if info.get('collision_occurred', False):
                collisions += 1

            # 存储经验
            for i in range(self.n_agents):
                if hasattr(self.memory, 'push'):
                    self.memory.push(states[i], actions[i], rewards[i], next_states[i], done)
                else:
                    self.memory.buffer.append((states[i], actions[i], rewards[i], next_states[i], done))

            # 更新网络
            if len(self.memory) >= self.config.training.batch_size and step % self.config.training.update_freq == 0:
                for i, agent in enumerate(self.agents):
                    loss_data = self._update_agent(agent, i)
                    if loss_data:
                        agent_losses['critic'][i] += loss_data[1]
                        agent_losses['policy'][i] += loss_data[2]
                        update_counts[i] += 1

            states = next_states

            if done:
                break

        # 计算平均损失
        avg_losses = {
            'critic': [losses / max(1, count) for losses, count in zip(agent_losses['critic'], update_counts)],
            'policy': [losses / max(1, count) for losses, count in zip(agent_losses['policy'], update_counts)]
        }

        return {
            'episode_reward': episode_reward,
            'steps': steps,
            'collisions': collisions,
            'success': info.get('reached_target', False),
            'formation': info.get('current_formation', 'unknown'),
            'avg_losses': avg_losses,
            'formation_error': self._calculate_formation_error()
        }

    def _update_agent(self, agent: ImprovedMASACAgent, agent_idx: int) -> Tuple[np.ndarray, float, float]:
        """更新单个智能体"""
        try:
            if hasattr(self.memory, 'sample'):
                # 优先经验回放
                batch_data = self.memory.sample(self.config.training.batch_size)
                if batch_data[0] is None:
                    return None

                (state_batch, action_batch, reward_batch, next_state_batch, done_batch), indices, weights = batch_data

                td_errors, critic_loss, policy_loss = agent.update(
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights
                )

                # 更新优先级
                if hasattr(self.memory, 'update_priorities') and indices is not None:
                    self.memory.update_priorities(indices, td_errors)

            else:
                # 简单经验回放
                import random
                batch = random.sample(list(self.memory.buffer), self.config.training.batch_size)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*batch))

                td_errors, critic_loss, policy_loss = agent.update(
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch
                )

            return td_errors, critic_loss, policy_loss

        except Exception as e:
            logger.error(f"智能体 {agent_idx} 更新失败: {e}")
            return None

    def _calculate_formation_error(self) -> float:
        """计算队形误差"""
        try:
            center = np.mean(self.env.positions, axis=0)
            desired_positions = self.env.formation_manager.get_desired_formation_positions(
                center, self.env.target_position
            )

            error = sum([
                np.linalg.norm(self.env.positions[i] - desired_positions[i])
                for i in range(self.n_agents)
            ])

            return error

        except Exception as e:
            logger.error(f"计算队形误差失败: {e}")
            return 0.0

    def _update_statistics(self, episode_stats: Dict[str, Any], episode: int):
        """更新训练统计 - 修复张量梯度警告"""
        self.episode_rewards.append(episode_stats['episode_reward'])
        self.success_rate.append(1 if episode_stats['success'] else 0)
        self.formation_errors.append(episode_stats['formation_error'])
        self.collision_counts.append(episode_stats['collisions'])

        # 记录到TensorBoard
        self.writer.add_scalar('Rewards/Episode_Reward', episode_stats['episode_reward'], episode)
        self.writer.add_scalar('Success/Success_Rate', episode_stats['success'], episode)
        self.writer.add_scalar('Errors/Formation_Error', episode_stats['formation_error'], episode)
        self.writer.add_scalar('Errors/Collision_Count', episode_stats['collisions'], episode)
        self.writer.add_scalar('Training/Steps_Per_Episode', episode_stats['steps'], episode)

        # 记录每个智能体的损失和参数 - 修复张量梯度警告
        for i in range(self.n_agents):
            self.writer.add_scalar(f'Agent_{i}/Critic_Loss', episode_stats['avg_losses']['critic'][i], episode)
            self.writer.add_scalar(f'Agent_{i}/Policy_Loss', episode_stats['avg_losses']['policy'][i], episode)

            # 安全地记录熵权重alpha - 修复张量梯度警告
            alpha_value = self._safe_get_alpha_value(self.agents[i])
            self.writer.add_scalar(f'Agent_{i}/Alpha', alpha_value, episode)

        # 记录队形类型
        self.writer.add_text('Formation', episode_stats['formation'], episode)

    def _safe_get_alpha_value(self, agent: ImprovedMASACAgent) -> float:
        """安全地获取alpha值，避免梯度警告"""
        try:
            if hasattr(agent.alpha, 'detach'):
                # 如果alpha是张量且需要梯度，先detach再转换
                return agent.alpha.detach().item()
            elif hasattr(agent.alpha, 'item'):
                # 如果alpha是不需要梯度的张量
                return agent.alpha.item()
            else:
                # 如果alpha是普通数值
                return float(agent.alpha)
        except Exception as e:
            logger.warning(f"获取alpha值失败: {e}, 使用默认值")
            return 0.2  # 返回默认alpha值

    def _log_progress(self, episode: int):
        """记录训练进度"""
        window = min(50, len(self.episode_rewards))
        avg_reward = np.mean(self.episode_rewards[-window:])
        avg_success = np.mean(self.success_rate[-window:]) * 100
        avg_formation_error = np.mean(self.formation_errors[-window:])
        avg_collisions = np.mean(self.collision_counts[-window:])

        # 记录移动平均值
        self.writer.add_scalar('Rewards/Average_Reward', avg_reward, episode)
        self.writer.add_scalar('Success/Success_Rate_Average', avg_success, episode)
        self.writer.add_scalar('Errors/Average_Formation_Error', avg_formation_error, episode)
        self.writer.add_scalar('Errors/Average_Collision_Count', avg_collisions, episode)

        logger.info(f"回合 {episode}/{self.config.training.n_episodes}")
        logger.info(f"  平均奖励: {avg_reward:.2f}")
        logger.info(f"  成功率: {avg_success:.1f}%")
        logger.info(f"  平均队形误差: {avg_formation_error:.2f}")
        logger.info(f"  平均碰撞次数: {avg_collisions:.2f}")
        logger.info(f"  缓冲区大小: {len(self.memory)}")

        # 安全地记录当前alpha值
        current_alpha = self._safe_get_alpha_value(self.agents[0])
        logger.info(f"  当前熵权重: {current_alpha:.4f}")

        # 检查最佳模型
        if avg_success > self.best_success_rate:
            self.best_success_rate = avg_success
            self.best_episode = episode
            self._save_best_model()
            logger.info(f"保存最佳模型 (回合 {episode}), 成功率: {avg_success:.1f}%")

    def _save_models(self, episode: int):
        """保存模型"""
        save_dir = f'models/episode_{episode}'
        os.makedirs(save_dir, exist_ok=True)

        for i, agent in enumerate(self.agents):
            agent.save_model(f'{save_dir}/agent_{i}.pth')

        logger.info(f"模型已保存到 {save_dir}")

    def _save_best_model(self):
        """保存最佳模型"""
        save_dir = 'models/best'
        os.makedirs(save_dir, exist_ok=True)

        for i, agent in enumerate(self.agents):
            agent.save_model(f'{save_dir}/agent_{i}.pth')

    def _evaluate_and_visualize(self, episode: int):
        """评估和可视化"""
        try:
            # 生成环境可视化
            fig = self.env.render(mode='return')
            if fig is not None:
                self.writer.add_figure('Environment', fig, episode)
                plt.close(fig)

            # 评估智能体性能
            eval_stats = self._evaluate_agents(n_episodes=5)
            logger.info(f"评估结果 (回合 {episode}): 成功率 {eval_stats['success_rate']:.1f}%, "
                        f"平均奖励 {eval_stats['avg_reward']:.2f}")

        except Exception as e:
            logger.error(f"评估和可视化失败: {e}")

    def _evaluate_agents(self, n_episodes: int = 5) -> Dict[str, float]:
        """评估智能体性能"""
        success_count = 0
        total_rewards = []

        for _ in range(n_episodes):
            states = self.env.reset()
            episode_reward = 0.0

            for step in range(self.config.environment.max_steps):
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.act(states[i], evaluate=True)
                    actions.append(action)

                next_states, rewards, done, info = self.env.step(actions)
                states = next_states
                episode_reward += sum(rewards)

                if done:
                    if info.get('reached_target', False):
                        success_count += 1
                    break

            total_rewards.append(episode_reward)

        return {
            'success_rate': success_count / n_episodes * 100,
            'avg_reward': np.mean(total_rewards)
        }

    def _should_early_stop(self, episode: int) -> bool:
        """检查是否应该早停"""
        if episode < 1000:
            return False

        recent_success_rate = np.mean(self.success_rate[-100:]) * 100

        # 如果成功率达到85%以上，可以考虑早停
        if recent_success_rate > 85:
            logger.info(f"达到满意的成功率 ({recent_success_rate:.1f}%)，考虑早停")
            return True

        return False

    def _cleanup(self):
        """清理资源"""
        try:
            # 保存最终模型
            save_dir = 'models/final'
            os.makedirs(save_dir, exist_ok=True)
            for i, agent in enumerate(self.agents):
                agent.save_model(f'{save_dir}/agent_{i}.pth')

            # 关闭环境
            if hasattr(self.env, 'close'):
                self.env.close()

            # 关闭TensorBoard
            self.writer.close()

            logger.info("训练资源清理完成")

        except Exception as e:
            logger.error(f"资源清理失败: {e}")

    def _get_training_summary(self) -> Dict[str, Any]:
        """获取训练总结"""
        training_duration = datetime.now() - self.training_start_time if self.training_start_time else None

        return {
            'episodes_completed': self.episodes_completed,
            'best_success_rate': self.best_success_rate,
            'best_episode': self.best_episode,
            'final_success_rate': np.mean(self.success_rate[-100:]) * 100 if len(self.success_rate) >= 100 else 0,
            'training_duration': training_duration,
            'episode_rewards': self.episode_rewards,
            'success_rate': self.success_rate,
            'formation_errors': self.formation_errors,
            'collision_counts': self.collision_counts
        }