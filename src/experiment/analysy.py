"""
analysis.py
信任博弈实验数据分析模块
功能：计算核心指标（基尼系数等）、存储绘图数据到Excel、生成论文风格图表
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import CommonUtils
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# 设置中文字体（避免图表中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrustGameAnalyzer:
    """
    信任博弈实验分析器
    支持加载多个CSV文件（新格式），计算基尼系数等核心指标，
    存储绘图数据到Excel，并生成论文风格的可视化图表。
    """

    # 论文图表风格设置
    STYLE = {
        "figure.figsize": (10, 6),
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
    }

    def __init__(self, csv_paths: List[str]):
        """
        初始化分析器，加载并合并多个CSV文件
        支持新格式字段：run_id, iteration, round, num_agents, proportion,
        agent_id, agent_type, 以及各类收益字段。
        """
        self.dfs = []
        for path in csv_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                self.dfs.append(df)
                logger.info(f"已加载数据: {path}, 行数: {len(df)}")
            else:
                logger.warning(f"文件不存在: {path}")

        if not self.dfs:
            raise ValueError("未加载到任何有效数据文件")

        self.df = pd.concat(self.dfs, ignore_index=True)

        # 确保数值列类型正确
        numeric_cols = ['run_id', 'iteration', 'round', 'num_agents', 'proportion', 'agent_id', 'agent_type',
                        'neighbor_1_id', 'neighbor_2_id', 'neighbor_3_id', 'neighbor_4_id',
                        'sent_to_n1', 'sent_to_n2', 'sent_to_n3', 'sent_to_n4', 'total_sent',
                        'received_return_from_n1', 'received_return_from_n2', 'received_return_from_n3', 'received_return_from_n4', 'total_received_return',
                        'received_send_from_n1', 'received_send_from_n2', 'received_send_from_n3', 'received_send_from_n4', 'total_received_send',
                        'returned_to_n1', 'returned_to_n2', 'returned_to_n3', 'returned_to_n4', 'total_returned',
                        'trustor_payoff', 'trustee_payoff', 'round_payoff', 'accumulate_payoff']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        logger.info(f"数据合并完成，总行数: {len(self.df)}，包含比例: {sorted(self.df['proportion'].unique())}")

        # 设置绘图风格
        sns.set_theme(style="whitegrid")
        plt.rcParams.update(self.STYLE)

    @staticmethod
    def gini_coefficient(values: np.ndarray) -> float:
        """计算基尼系数（仅非负值）"""
        values = values[values >= 0]
        if len(values) == 0 or np.sum(values) == 0:
            return 0.0
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        # Gini = (2 * Σ(i * xi)) / (n * Σxi) - (n+1)/n
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_vals)) - (n+1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals))
        return gini

    def compute_core_metrics(self, proportion: Optional[float] = None) -> Dict:
        """
        计算核心指标，可指定 proportion 筛选
        返回:
            - mean_invested: 平均委托金额 (total_sent / 4)
            - mean_returned: 平均返还金额 (total_returned / 4)
            - total_payoff_sum: 群体总收益（单轮总收益之和，或累加收益）
            - agent_final_payoffs: 每个智能体的最终累计收益（最后一轮的 accumulate_payoff）
            - gini: 基尼系数
            - round_metrics: 按轮次聚合的DataFrame（平均委托、返还、群体总收益）
        """
        df = self.df if proportion is None else self.df[self.df['proportion'] == proportion]
        if len(df) == 0:
            raise ValueError(f"未找到 proportion={proportion} 的数据")

        # 平均委托/返还金额（按交互记录）
        # total_sent 是发送给4个邻居的总和，取平均每次发送
        mean_invested = df['total_sent'].mean() / 4
        # total_returned 是从4个邻居收到的返还总和，取平均每次返还
        mean_returned = df['total_returned'].mean() / 4

        # 群体总收益：所有 agent 在最后一轮的 accumulate_payoff 之和
        # 每个 agent 的最终累计收益（取每个 agent 的最大 round 对应的 accumulate_payoff）
        final_payoffs = df.sort_values('round').groupby('agent_id')['accumulate_payoff'].last()
        total_payoff_sum = final_payoffs.sum()

        # 基尼系数基于最终累计收益
        gini = self.gini_coefficient(final_payoffs.values)

        # 按轮次聚合：平均委托、平均返还、每轮所有 agent 的 round_payoff 之和（群体单轮收益）
        round_metrics = df.groupby('round').agg(
            mean_invested=('total_sent', lambda x: x.mean() / 4),
            mean_returned=('total_returned', lambda x: x.mean() / 4),
            total_payoff=('round_payoff', 'sum')
        ).reset_index()

        return {
            'mean_invested': mean_invested,
            'mean_returned': mean_returned,
            'total_payoff_sum': total_payoff_sum,
            'agent_final_payoffs': final_payoffs,
            'gini': gini,
            'round_metrics': round_metrics,
        }

    def export_plot_data_to_excel(self, output_excel_path: str):
        """
        将绘图所需的核心数据存储到 Excel 的一个工作簿中。
        包含多个工作表：
            - summary: 各比例下的平均委托、返还、基尼系数、总收益
            - round_mean_invested: 各比例每轮平均委托金额（透视表）
            - round_mean_returned: 各比例每轮平均返还金额（透视表）
            - round_total_payoff: 各比例每轮群体总收益（透视表）
            - gini_by_proportion: 各比例基尼系数
        """
        proportions = sorted(self.df['proportion'].unique())
        summary_data = []
        round_invested_pivot = None
        round_returned_pivot = None
        round_payoff_pivot = None
        gini_list = []

        for prop in proportions:
            metrics = self.compute_core_metrics(proportion=prop)
            summary_data.append({
                'proportion': prop,
                'mean_invested': metrics['mean_invested'],
                'mean_returned': metrics['mean_returned'],
                'total_payoff_sum': metrics['total_payoff_sum'],
                'gini': metrics['gini']
            })
            gini_list.append({'proportion': prop, 'gini': metrics['gini']})

            # 构建轮次透视表
            round_df = metrics['round_metrics']
            round_df['proportion'] = prop
            # 合并各比例的轮次数据
            if round_invested_pivot is None:
                round_invested_pivot = round_df[['round', 'proportion', 'mean_invested']]
                round_returned_pivot = round_df[['round', 'proportion', 'mean_returned']]
                round_payoff_pivot = round_df[['round', 'proportion', 'total_payoff']]
            else:
                round_invested_pivot = pd.concat([round_invested_pivot, round_df[['round', 'proportion', 'mean_invested']]])
                round_returned_pivot = pd.concat([round_returned_pivot, round_df[['round', 'proportion', 'mean_returned']]])
                round_payoff_pivot = pd.concat([round_payoff_pivot, round_df[['round', 'proportion', 'total_payoff']]])

        # 透视表：行=round，列=proportion，值=指标
        if round_invested_pivot is not None:
            round_invested_table = round_invested_pivot.pivot(index='round', columns='proportion', values='mean_invested')
            round_returned_table = round_returned_pivot.pivot(index='round', columns='proportion', values='mean_returned')
            round_payoff_table = round_payoff_pivot.pivot(index='round', columns='proportion', values='total_payoff')
            # 重命名列
            round_invested_table.columns = [f'proportion={c}' for c in round_invested_table.columns]
            round_returned_table.columns = [f'proportion={c}' for c in round_returned_table.columns]
            round_payoff_table.columns = [f'proportion={c}' for c in round_payoff_table.columns]
            round_invested_table.reset_index(inplace=True)
            round_returned_table.reset_index(inplace=True)
            round_payoff_table.reset_index(inplace=True)

        summary_df = pd.DataFrame(summary_data)
        gini_df = pd.DataFrame(gini_list)

        # 写入 Excel
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='summary', index=False)
            if round_invested_table is not None:
                round_invested_table.to_excel(writer, sheet_name='round_mean_invested', index=False)
                round_returned_table.to_excel(writer, sheet_name='round_mean_returned', index=False)
                round_payoff_table.to_excel(writer, sheet_name='round_total_payoff', index=False)
            gini_df.to_excel(writer, sheet_name='gini_by_proportion', index=False)

        logger.info(f"绘图数据已导出至: {output_excel_path}")
        return output_excel_path

    def plot_figures(self, output_dir: str):
        """
        生成论文同款图表：
        - 图1: 委托金额随轮次变化（不同比例对比）
        - 图2: 返还金额随轮次变化
        - 图3: 基尼系数条形图
        - 图4: 个人最终累计收益分布箱线图
        """
        proportions = sorted(self.df['proportion'].unique())
        os.makedirs(output_dir, exist_ok=True)

        colors = sns.color_palette("Blues_d", n_colors=len(proportions))
        prop_labels = [f"{int(p*100)}%" for p in proportions]

        # 图1：委托金额随轮次变化
        plt.figure(figsize=(10, 6))
        for i, prop in enumerate(proportions):
            metrics = self.compute_core_metrics(proportion=prop)
            round_df = metrics['round_metrics']
            plt.plot(round_df['round'], round_df['mean_invested'],
                     marker='o', markersize=4, linewidth=1.5,
                     color=colors[i], label=f"自由{prop_labels[i]}")
        plt.xlabel("轮次")
        plt.ylabel("平均委托金额")
        plt.title("不同自由玩家比例下的平均委托金额演化")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "invested_over_rounds.png"), dpi=150)
        plt.close()

        # 图2：返还金额随轮次变化
        plt.figure(figsize=(10, 6))
        for i, prop in enumerate(proportions):
            metrics = self.compute_core_metrics(proportion=prop)
            round_df = metrics['round_metrics']
            plt.plot(round_df['round'], round_df['mean_returned'],
                     marker='s', markersize=4, linewidth=1.5,
                     color=colors[i], label=f"自由{prop_labels[i]}")
        plt.xlabel("轮次")
        plt.ylabel("平均返还金额")
        plt.title("不同自由玩家比例下的平均返还金额演化")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "returned_over_rounds.png"), dpi=150)
        plt.close()

        # 图3：基尼系数条形图
        ginis = []
        for prop in proportions:
            metrics = self.compute_core_metrics(proportion=prop)
            ginis.append(metrics['gini'])
        plt.figure(figsize=(8, 5))
        bars = plt.bar(prop_labels, ginis, color=colors, edgecolor='black')
        plt.xlabel("自由玩家比例")
        plt.ylabel("基尼系数")
        plt.title("不同自由玩家比例下的财富不平等程度")
        y_max = max(ginis) * 1.2 if max(ginis) > 0 else 0.6
        plt.ylim(0, y_max)
        for bar, g in zip(bars, ginis):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{g:.3f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gini_coefficient.png"), dpi=150)
        plt.close()

        # 图4：个人最终累计收益分布箱线图
        plt.figure(figsize=(10, 6))
        data_to_plot = []
        for prop in proportions:
            metrics = self.compute_core_metrics(proportion=prop)
            agent_payoffs = metrics['agent_final_payoffs'].values
            data_to_plot.append(agent_payoffs)
        plt.boxplot(data_to_plot, labels=prop_labels, showmeans=True, meanline=True)
        plt.xlabel("自由玩家比例")
        plt.ylabel("个人累计收益")
        plt.title("不同自由玩家比例下的个人收益分布")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "personal_payoff_boxplot.png"), dpi=150)
        plt.close()

        logger.info(f"所有图表已保存至: {output_dir}")


# 命令行使用示例
if __name__ == "__main__":
    import glob

    is_multi_file = True
    # 默认 CSV 文件路径
    csv_dir = os.path.join(
        CommonUtils.get_project_root_path(),
        "outputs",
        "20260430_220005_deepseek-v4-flash_trust_game"
    )

    if is_multi_file:
        import glob
        # 使用通配符匹配所有CSV文件
        pattern = os.path.join(csv_dir, "*.csv")
        csv_files = glob.glob(pattern)
        if csv_files:
            print(f"找到 {len(csv_files)} 个CSV文件:")
            for f in csv_files:
                print(f"  - {f}")

            analyzer = TrustGameAnalyzer(csv_files)

    else:
        file_name = "20260422_112210_trust_game_p-[0, 0.25, 0.5, 0.75, 1].csv"
        file_path = os.path.join(csv_dir, file_name)

        if os.path.exists(file_path):
            analyzer = TrustGameAnalyzer([file_path])
            # 导出绘图数据到Excel
            analyzer.export_plot_data_to_excel(os.path.join(csv_dir, "plot_data.xlsx"))
            # 生成图表
            analyzer.plot_figures(os.path.join(csv_dir, "figures"))

            print("分析完成！")
        else:
            print(f"文件不存在: {file_path}")