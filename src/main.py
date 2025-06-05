"""
改进的多智能体强化学习主程序 - Windows兼容版本
"""
import os
import sys
import logging
import argparse
from datetime import datetime

# 添加源码路径
sys.path.append('src')

from config import Config
from trainer import ImprovedTrainer
from utils import setup_logging, create_visualization, save_training_results, safe_log_message


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='改进的多智能体强化学习训练')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的模型路径')
    parser.add_argument('--evaluate', action='store_true',
                        help='仅评估模式，不进行训练')
    parser.add_argument('--render', action='store_true',
                        help='启用实时渲染')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()

    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(safe_log_message("🚀 改进的多智能体强化学习训练开始"))
    logger.info(safe_log_message(f"⏰ 开始时间: {datetime.now()}"))
    logger.info("=" * 60)

    try:
        # 加载配置
        config = Config.from_yaml(args.config)
        logger.info(safe_log_message(f"✅ 配置加载完成: {args.config}"))

        if args.debug:
            # 调试模式下使用较小的配置
            config.training.n_episodes = 100
            config.training.batch_size = 32
            config.training.buffer_size = 10000
            logger.info(safe_log_message("🐛 调试模式：使用较小的训练配置"))

        # 创建训练器
        trainer = ImprovedTrainer(config)
        logger.info(safe_log_message("✅ 训练器创建完成"))

        if args.resume:
            # 恢复训练
            logger.info(safe_log_message(f"🔄 从 {args.resume} 恢复训练"))
            # trainer.load_models(args.resume)  # 这个方法需要实现

        if args.evaluate:
            # 评估模式
            logger.info(safe_log_message("🧪 开始评估模式"))
            eval_results = trainer._evaluate_agents(n_episodes=10)
            logger.info(safe_log_message(f"📊 评估结果: {eval_results}"))
        else:
            # 训练模式
            logger.info(safe_log_message("🏋️ 开始训练"))
            training_results = trainer.train()

            # 保存训练结果
            save_training_results(training_results, config)

            # 创建可视化
            create_visualization(training_results)

            logger.info("=" * 60)
            logger.info(safe_log_message("🎉 训练完成!"))
            logger.info(safe_log_message(f"📈 最佳成功率: {training_results['best_success_rate']:.1f}%"))
            logger.info(safe_log_message(f"🏆 最佳回合: {training_results['best_episode']}"))
            logger.info(safe_log_message(f"⏱️ 训练时长: {training_results['training_duration']}"))
            logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info(safe_log_message("❌ 训练被用户中断"))
    except Exception as e:
        logger.error(safe_log_message(f"💥 训练过程中出现错误: {e}"))
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()