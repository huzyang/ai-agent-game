# batch_run.py - 对称信任博弈批量仿真
import mesa
import pandas as pd
import os
from datetime import datetime
from params import Params
from model import GameModel

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

def run_batch_experiment(params):
    """
    批量运行实验
    """
    params.print_all_params()

    # 运行批量实验
    print(f"\n开始批量运行...")
    import time
    start_time = time.time()

    results = mesa.batch_run(
        GameModel,
        parameters=params.model_init_params,
        iterations=params.iterations,
        max_steps=params.max_steps,
        number_processes=params.number_processes,
        data_collection_period=params.data_collection_period,
        display_progress=True,
    )

    run_time = time.time() - start_time
    print(f"批量运行完成，耗时: {run_time:.2f}秒")
    print(f"总共运行次数: {len(results)}")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CommonUtils.get_project_root_path(),"outputs", f"{timestamp}_batch_results")
    os.makedirs(output_dir, exist_ok=True)

    # 转换为DataFrame并保存
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, f"{timestamp}_all_results.csv"), index=False)

    # 保存运行摘要
    summary = f"""LLM multi Agent 实验摘要
    ==========================================
    运行时间: {timestamp}
    运行耗时: {format_run_time(run_time)}
    参数配置:
    {params.record_params()}
    ==========================================
    """
    # {json.dumps(params.record_params(), indent=2, ensure_ascii=False)}
    with open(os.path.join(output_dir, f"{timestamp}_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"结果已保存到: {output_dir}")

    return results, output_dir, timestamp

def main():
    """主函数"""
    # 定义批量运行参数
    params = Params()

    # 运行批量实验
    results, output_dir, timestamp = run_batch_experiment(params=params)

    # 结果分析与保存

    # 分析结果并生成可视化


if __name__ == "__main__":
    # 注意：由于多进程问题，在Windows上运行可能需要将代码放在if __name__ == "__main__":块中
    # 导入CommonUtils
    from src.utils import CommonUtils

    main()