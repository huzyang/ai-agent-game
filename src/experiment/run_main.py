"""
阶段2主运行脚本：分别运行囚徒困境、信任博弈、最后通牒博弈各50轮，
使用两个智能体（可配置为受限或自由），验证基本功能。
"""

import os
import sys
import logging
from datetime import datetime

from src.experiment.params import Params
from src.experiment.model import GameModel
# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def setup_logging():
    """设置日志输出到文件和控制台"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"stage2_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()
    print("=" * 60)
    print("LLM智能体博弈实验")
    print("=" * 60)

    params = Params()
    params.print_all_params()

    model = GameModel(params.width, params.height, params.game_type)
    model.run_model(params.rounds)

if __name__ == "__main__":
    main()