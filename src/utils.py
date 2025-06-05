"""
工具函数 - Windows兼容版本
"""
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any
import yaml
import sys


def setup_logging(level=logging.INFO):
    """设置Windows兼容的日志系统"""
    os.makedirs('logs', exist_ok=True)

    # 为Windows系统设置UTF-8编码
    if sys.platform.startswith('win'):
        # Windows系统特殊处理
        import codecs

        # 创建日志格式（不使用emoji）
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 文件处理器（使用UTF-8编码）
        log_filename = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)

        # 控制台处理器（使用UTF-8编码）
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)

        # 尝试设置控制台编码
        try:
            # 在Windows上设置控制台编码为UTF-8
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)  # UTF-8
            console_handler.stream = codecs.getwriter('utf-8')(console_handler.stream.buffer)
        except:
            # 如果设置失败，使用默认编码
            pass
    else:
        # Unix/Linux系统
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        file_handler = logging.FileHandler(
            f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def safe_log_message(message: str) -> str:
    """安全的日志消息（移除或替换emoji）"""
    if sys.platform.startswith('win'):
        # Windows系统：替换emoji为文字
        emoji_map = {
            '🚀': '[START]',
            '⏰': '[TIME]',
            '✅': '[OK]',
            '🏋️': '[TRAIN]',
            '🎉': '[COMPLETE]',
            '📈': '[CHART]',
            '🏆': '[BEST]',
            '⏱️': '[DURATION]',
            '❌': '[ERROR]',
            '💥': '[CRASH]',
            '🧪': '[TEST]',
            '📊': '[STATS]',
            '🔄': '[RESUME]',
            '🐛': '[DEBUG]'
        }

        for emoji, replacement in emoji_map.items():
            message = message.replace(emoji, replacement)

    return message


def save_training_results(results: Dict[str, Any], config):
    """保存训练结果"""
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results/{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    # 保存配置
    config_dict = {
        'training': config.training.__dict__,
        'environment': config.environment.__dict__,
        'parallel': config.parallel.__dict__
    }

    with open(f'{results_dir}/config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    # 保存训练结果
    results_to_save = results.copy()
    # 转换numpy数组为列表以便JSON序列化
    for key, value in results_to_save.items():
        if isinstance(value, np.ndarray):
            results_to_save[key] = value.tolist()
        elif hasattr(value, 'total_seconds'):  # datetime.timedelta
            results_to_save[key] = str(value)

    with open(f'{results_dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)

    logging.getLogger(__name__).info(f"训练结果已保存到: {results_dir}")


def create_visualization(results: Dict[str, Any]):
    """创建训练可视化"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('训练结果可视化', fontsize=16)

    # 奖励曲线
    axes[0, 0].plot(results['episode_rewards'])
    axes[0, 0].set_title('回合奖励')
    axes[0, 0].set_xlabel('回合')
    axes[0, 0].set_ylabel('奖励')
    axes[0, 0].grid(True)

    # 成功率曲线
    window_size = min(100, len(results['success_rate']))
    if window_size > 0:
        success_rate_ma = np.convolve(
            results['success_rate'],
            np.ones(window_size) / window_size,
            mode='valid'
        )
        axes[0, 1].plot(success_rate_ma)
    axes[0, 1].set_title('成功率 (移动平均)')
    axes[0, 1].set_xlabel('回合')
    axes[0, 1].set_ylabel('成功率')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True)

    # 队形误差
    axes[1, 0].plot(results['formation_errors'])
    axes[1, 0].set_title('队形误差')
    axes[1, 0].set_xlabel('回合')
    axes[1, 0].set_ylabel('误差')
    axes[1, 0].grid(True)

    # 碰撞次数
    axes[1, 1].plot(results['collision_counts'])
    axes[1, 1].set_title('碰撞次数')
    axes[1, 1].set_xlabel('回合')
    axes[1, 1].set_ylabel('碰撞次数')
    axes[1, 1].grid(True)

    plt.tight_layout()

    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/training_visualization_{timestamp}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    logging.getLogger(__name__).info(f"可视化图表已保存")


def monitor_system_resources():
    """监控系统资源"""
    try:
        import psutil

        # CPU和内存
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # GPU (如果有)
        gpu_info = ""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_info = f", GPU: {gpu.load * 100:.1f}% (Memory: {gpu.memoryUtil * 100:.1f}%)"
        except:
            pass

        logger = logging.getLogger(__name__)
        logger.debug(f"系统资源 - CPU: {cpu_percent:.1f}%, "
                     f"Memory: {memory.percent:.1f}%{gpu_info}")

        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available': memory.available
        }

    except ImportError:
        return None


def estimate_training_time(current_episode: int, total_episodes: int,
                           start_time: datetime) -> str:
    """估算剩余训练时间"""
    if current_episode == 0:
        return "估算中..."

    elapsed = datetime.now() - start_time
    time_per_episode = elapsed.total_seconds() / current_episode
    remaining_episodes = total_episodes - current_episode
    estimated_remaining = remaining_episodes * time_per_episode

    hours, remainder = divmod(estimated_remaining, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"