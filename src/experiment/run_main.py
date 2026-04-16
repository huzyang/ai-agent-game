"""
阶段2主运行脚本：分别运行囚徒困境、信任博弈、最后通牒博弈各50轮，
使用两个智能体（可配置为受限或自由），验证基本功能。
"""

import os
import sys
import logging
import tqdm
import pandas as pd
from datetime import datetime
from src.experiment.params import Params,GameScenario
from src.experiment.model import GameModel
from src.utils import CommonUtils

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger('experiment')
logger.setLevel(logging.INFO)
def setup_logging(output_dir):
    """设置日志输出到文件和控制台"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"run.log")

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="a", encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False


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

def multi_round_exp(params: Params,output_dir):
    """
    执行多轮实验的主函数，针对给定的模型列表进行多轮测试
    参数:
        model_list: 要测试的模型或模型列表
        exp_time: 每个模型要重复实验的次数
        round_num_inform: 是否在输出中显示轮次信息
    """
    model_type_list = params.model_type_list
    iterations = params.iterations
    total_iterations = len(model_type_list) * iterations
    with tqdm.tqdm(total=total_iterations, desc="Overall Progress") as pbar:
        for model_type in model_type_list:
            for iteration in range(1, iterations + 1):
                # 执行多轮实验，传入模型、角色列表、文件夹路径等参数
                game_model = GameModel(scenario=GameScenario())
                all_results = game_model.run_model(params.rounds)

                filename = f"{params.game_type}_p-{params.proportion}_{iteration}.csv"
                filepath = os.path.join(output_dir, filename)

                df = pd.DataFrame(all_results)
                df.to_csv(filepath, index=False, encoding='utf-8')
                logger.info(f"使用{model_type}大模型，第{iteration}次重复运行的实验结果已保存到 {filepath}")

                pbar.update(1)
                pbar.set_postfix({
                    'model': model_type.value if hasattr(model_type, 'value') else model_type,
                    'iteration': iteration
                })



def main():
    params = Params()
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CommonUtils.get_project_root_path(), "outputs", f"{timestamp}_{params.model_type}_{params.game_type}")
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("开始运行LLM智能体博弈实验...")
    logger.info("=" * 60)

    import time
    start_time = time.time()
    multi_round_exp(params=params, output_dir=output_dir)

    run_time = format_run_time(time.time() - start_time)
    logger.info(f"所有LLM智能体博弈实验运行完成，共耗时: {run_time}")


if __name__ == "__main__":
    main()
