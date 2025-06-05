"""
配置管理模块
"""
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional
import os


@dataclass
class TrainingConfig:
    """训练配置"""
    n_episodes: int = 2000
    batch_size: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    buffer_size: int = int(3e5)
    update_freq: int = 4
    auto_entropy_tuning: bool = True
    use_prioritized_replay: bool = True
    hidden_dim: int = 256
    max_grad_norm: float = 0.5


@dataclass
class EnvironmentConfig:
    """环境配置"""
    n_agents: int = 3
    world_size: int = 20
    n_static_obstacles: int = 5
    n_dynamic_obstacles: int = 2
    max_steps: int = 200
    agent_radius: float = 0.5
    obstacle_radius: float = 1.0


@dataclass
class ParallelConfig:
    """并行训练配置"""
    n_envs: int = 4
    use_parallel: bool = True
    max_workers: int = 4


@dataclass
class Config:
    """主配置类"""
    training: TrainingConfig
    environment: EnvironmentConfig
    parallel: ParallelConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """从YAML文件加载配置"""
        if not os.path.exists(config_path):
            # 创建默认配置文件
            cls.create_default_config(config_path)

        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return cls(
            training=TrainingConfig(**data.get('training', {})),
            environment=EnvironmentConfig(**data.get('environment', {})),
            parallel=ParallelConfig(**data.get('parallel', {}))
        )

    @classmethod
    def create_default_config(cls, config_path: str):
        """创建默认配置文件"""
        default_config = {
            'training': {
                'n_episodes': 2000,
                'batch_size': 512,
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'alpha': 0.2,
                'buffer_size': 300000,
                'update_freq': 4,
                'auto_entropy_tuning': True,
                'use_prioritized_replay': True,
                'hidden_dim': 256,
                'max_grad_norm': 0.5
            },
            'environment': {
                'n_agents': 3,
                'world_size': 20,
                'n_static_obstacles': 5,
                'n_dynamic_obstacles': 2,
                'max_steps': 200,
                'agent_radius': 0.5,
                'obstacle_radius': 1.0
            },
            'parallel': {
                'n_envs': 4,
                'use_parallel': True,
                'max_workers': 4
            }
        }

        os.makedirs(os.path.dirname(config_path) if os.path.dirname(config_path) else '.', exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)