"""
阶段2主运行脚本：分别运行囚徒困境、信任博弈、最后通牒博弈各50轮，
使用两个智能体（可配置为受限或自由），验证基本功能。
"""

import os
import sys
import logging
import tqdm
from datetime import datetime

from src.experiment.params import Params
from src.experiment.model import GameModel
from src.utils import CommonUtils
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

def format_run_time(seconds):
    """
    自动将秒数转换为最合适的单位（时、分、秒）

    参数:
    - seconds: 运行时间（秒）

    返回:
    - 格式化的时间字符串
    """
    if seconds >= 3600:  # 大于等于1小时
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}小时{minutes}分{secs:.2f}秒"
    elif seconds >= 60:  # 大于等于1分钟
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.2f}秒"
    else:  # 小于1分钟
        return f"{seconds:.2f}秒"

def multi_round_exp(
        params  # 参数对象
):
    """
    执行多轮实验的主函数，针对给定的模型列表进行多轮测试
    参数:
        model_list: 要测试的模型或模型列表
        exp_time: 每个模型要重复实验的次数
        round_num_inform: 是否在输出中显示轮次信息
    """
    model_list = params.model_list
    iterations = params.iterations
    for model in model_list:

        # 根据模型列表创建结果文件夹
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(CommonUtils.get_project_root_path(), "outputs", f"{timestamp}_{params.run_type}_{params.analyzed_params[0]}-{params.analyzed_params[1]}")
        os.makedirs(output_dir, exist_ok=True)

        # 生成初始设置，获取文件夹路径和额外的提示信息

        # 使用t进度条进行多轮实验
        for i in tqdm.trange(iterations):
            # 执行多轮实验，传入模型、角色列表、文件夹路径等参数
            game_model = GameModel(params.width, params.height, params.game_type)
            game_model.run_model(params.rounds)


def main():
    logger = setup_logging()
    print("=" * 60)
    print("开始运行LLM智能体博弈实验...")
    print("=" * 60)

    params = Params()
    params.print_all_params()

    import time
    start_time = time.time()
    multi_round_exp(params)


    run_time = format_run_time(time.time() - start_time)
    print(f"批量运行完成，耗时: {run_time:.2f}秒")


if __name__ == "__main__":
    main()