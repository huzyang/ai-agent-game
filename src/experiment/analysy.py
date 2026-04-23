"""
analysis.py
信任博弈实验数据分析模块
功能：计算核心指标、行为聚类、统计检验、生成论文风格图表
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from typing import List, Dict, Optional
import logging

from src.utils import CommonUtils

logger = logging.getLogger(__name__)

# 设置中文字体（避免图表中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrustGameAnalyzer:
    """
    信任博弈实验分析器
    支持加载多个CSV文件（不同 proportion 或重复实验），计算核心指标，
    进行行为表型聚类和统计检验，并生成论文风格的可视化图表。
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
        Args:
            csv_paths: CSV文件路径列表，每个文件应包含字段：
                run_id, iteration, round, num_agents, proportion,
                investor_agent_id, investor_agent_type, trustee_neighbor_id,
                invested_amount, returned_amount, investor_agent_payoff, trustee_agent_payoff
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
        numeric_cols = ['invested_amount', 'returned_amount', 'investor_agent_payoff', 'trustee_agent_payoff']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        # 添加辅助列：每笔交互的总收益（投资者+受托者）
        # self.df['interaction_total_payoff'] = self.df['investor_agent_payoff'] + self.df['trustee_agent_payoff']

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
        Returns:
            字典包含:
                - mean_invested: 平均委托金额
                - mean_returned: 平均返还金额
                - total_payoff_sum: 群体总收益（所有交互的投资者+受托者收益之和）
                - agent_total_payoffs: 每个智能体的累计收益（Series）
                - gini: 基尼系数
                - round_metrics: 按轮次聚合的DataFrame（平均委托、返还、群体总收益）
        """
        df = self.df if proportion is None else self.df[self.df['proportion'] == proportion]
        if len(df) == 0:
            raise ValueError(f"未找到 proportion={proportion} 的数据")

        # 平均委托/返还金额（按交互记录）
        mean_invested = df['invested_amount'].mean()
        mean_returned = df['returned_amount'].mean()

        # 群体总收益
        total_payoff_sum = df['interaction_total_payoff'].sum()

        # 个人累计收益
        # 作为投资者时的收益
        investor_sum = df.groupby('investor_agent_id')['investor_agent_payoff'].sum()
        # 作为受托者时的收益
        trustee_sum = df.groupby('trustee_neighbor_id')['trustee_agent_payoff'].sum()
        agent_total = investor_sum.add(trustee_sum, fill_value=0).sort_index()
        gini = self.gini_coefficient(agent_total.values)

        # 按轮次聚合
        round_metrics = df.groupby('round').agg(
            mean_invested=('invested_amount', 'mean'),
            mean_returned=('returned_amount', 'mean'),
            total_payoff=('interaction_total_payoff', 'sum')
        ).reset_index()

        return {
            'mean_invested': mean_invested,
            'mean_returned': mean_returned,
            'total_payoff_sum': total_payoff_sum,
            'agent_total_payoffs': agent_total,
            'gini': gini,
            'round_metrics': round_metrics,
        }

    def behavior_clustering(self, proportion: Optional[float] = None, n_clusters: int = 3) -> pd.DataFrame:
        """
        行为表型聚类（基于平均委托金额和平均返还金额）
        输出三类：prosocial（亲社会）, neutral（中性）, antisocial（反社会）
        分类依据：聚类后按平均委托金额排序，高 -> prosocial，中 -> neutral，低 -> antisocial
        Returns:
            DataFrame 包含每个 investor_agent_id 及其聚类标签、特征值、表型
        """
        df = self.df if proportion is None else self.df[self.df['proportion'] == proportion]
        if len(df) == 0:
            raise ValueError(f"未找到 proportion={proportion} 的数据")

        # 提取每个 agent 作为投资者的平均行为
        agent_investor = df.groupby('investor_agent_id').agg(
            mean_invested=('invested_amount', 'mean'),
            mean_returned=('returned_amount', 'mean')
        ).reset_index()

        # 标准化
        features = agent_investor[['mean_invested', 'mean_returned']].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # K-means 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        agent_investor['cluster'] = labels

        # 根据平均委托金额排序确定标签含义
        cluster_order = agent_investor.groupby('cluster')['mean_invested'].mean().sort_values().index.tolist()
        if len(cluster_order) == 3:
            type_map = {
                cluster_order[0]: 'antisocial',
                cluster_order[1]: 'neutral',
                cluster_order[2]: 'prosocial'
            }
        elif len(cluster_order) == 2:
            type_map = {cluster_order[0]: 'antisocial', cluster_order[1]: 'prosocial'}
        else:
            type_map = {c: f'type_{c}' for c in cluster_order}

        agent_investor['phenotype'] = agent_investor['cluster'].map(type_map)
        logger.info(f"行为聚类完成，各类别计数:\n{agent_investor['phenotype'].value_counts()}")
        return agent_investor

    def t_test_analysis(self, group1_proportion: float, group2_proportion: float, metric: str = 'invested_amount') -> Dict:
        """
        对两组不同 proportion 的实验进行独立样本 t 检验 (Welch's t-test)
        Args:
            group1_proportion: 第一组的自由玩家比例
            group2_proportion: 第二组的自由玩家比例
            metric: 比较的指标，可选 'invested_amount', 'returned_amount', 'investor_agent_payoff'
        Returns:
            包含 t 统计量、p 值、均值等的字典
        """
        df1 = self.df[self.df['proportion'] == group1_proportion]
        df2 = self.df[self.df['proportion'] == group2_proportion]
        if len(df1) == 0 or len(df2) == 0:
            raise ValueError("指定的比例在数据中不存在")

        if metric == 'invested_amount':
            vals1 = df1.groupby('investor_agent_id')['invested_amount'].mean().dropna()
            vals2 = df2.groupby('investor_agent_id')['invested_amount'].mean().dropna()
        elif metric == 'returned_amount':
            vals1 = df1.groupby('investor_agent_id')['returned_amount'].mean().dropna()
            vals2 = df2.groupby('investor_agent_id')['returned_amount'].mean().dropna()
        elif metric == 'investor_agent_payoff':
            vals1 = df1.groupby('investor_agent_id')['investor_agent_payoff'].sum().dropna()
            vals2 = df2.groupby('investor_agent_id')['investor_agent_payoff'].sum().dropna()
        else:
            raise ValueError("metric 必须是 'invested_amount', 'returned_amount' 或 'investor_agent_payoff'")

        t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False)
        return {
            't_statistic': t_stat,
            'p_value': p_val,
            'mean_group1': vals1.mean(),
            'mean_group2': vals2.mean(),
            'metric': metric,
            'group1_proportion': group1_proportion,
            'group2_proportion': group2_proportion
        }

    def generate_report(self, output_path: str, proportions: List[float] = None):
        """
        生成文本分析报告
        Args:
            output_path: 报告保存路径
            proportions: 要分析的 proportion 列表，默认为数据中所有的唯一比例
        """
        if proportions is None:
            proportions = sorted(self.df['proportion'].unique())

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("信任博弈实验分析报告\n")
            f.write("=" * 70 + "\n\n")

            for prop in proportions:
                f.write(f"【自由玩家比例: {prop:.0%}】\n")
                metrics = self.compute_core_metrics(proportion=prop)
                f.write(f"  平均委托金额: {metrics['mean_invested']:.3f}\n")
                f.write(f"  平均返还金额: {metrics['mean_returned']:.3f}\n")
                f.write(f"  群体总收益: {metrics['total_payoff_sum']:.2f}\n")
                f.write(f"  基尼系数: {metrics['gini']:.4f}\n\n")

            if len(proportions) >= 2:
                f.write("【统计检验结果】\n")
                if 0.0 in proportions and 1.0 in proportions:
                    res = self.t_test_analysis(0.0, 1.0, metric='invested_amount')
                    f.write(f"  委托金额 t检验 (0%自由 vs 100%自由): t={res['t_statistic']:.3f}, p={res['p_value']:.4f}\n")
                    res2 = self.t_test_analysis(0.0, 1.0, metric='returned_amount')
                    f.write(f"  返还金额 t检验 (0%自由 vs 100%自由): t={res2['t_statistic']:.3f}, p={res2['p_value']:.4f}\n")

            f.write("\n【行为表型聚类结果】\n")
            for prop in proportions:
                clustering = self.behavior_clustering(proportion=prop, n_clusters=3)
                counts = clustering['phenotype'].value_counts()
                f.write(f"  比例 {prop:.0%}:\n")
                for pheno, cnt in counts.items():
                    f.write(f"    {pheno}: {cnt} 人 ({cnt/len(clustering)*100:.1f}%)\n")
                f.write("\n")

        logger.info(f"分析报告已保存至: {output_path}")

    def plot_returned_amount_with_effect_size(self, output_path: str, reference_proportion: float = 0.0):
        """
        生成与参考图片风格一致的组合图：
          上半部分：返还金额的箱线图 + 均值点图（含置信区间）
          下半部分：效应量（相对于参考比例）点图（含置信区间）

        Args:
            output_path: 图像保存路径（如 .png 或 .pdf）
            reference_proportion: 作为参考的自由玩家比例（默认 0.0）
        """
        # 获取所有比例（排序）
        proportions = sorted(self.df['proportion'].unique())
        # 确保参考比例在列表中
        if reference_proportion not in proportions:
            raise ValueError(f"参考比例 {reference_proportion} 不在数据中")

        # 1. 按比例 + agent 聚合平均返还金额
        # 注意：每个 agent 在同一个比例下可能有多个轮次和多个邻居，取平均作为该 agent 的代表值
        agent_avg_return = self.df.groupby(['proportion', 'investor_agent_id'])['returned_amount'].mean().reset_index()

        # 存储每个比例下的所有个体平均返还金额（用于箱线图）
        data_for_box = []
        for p in proportions:
            vals = agent_avg_return[agent_avg_return['proportion'] == p]['returned_amount'].values
            data_for_box.append(vals)

        # 2. 计算每个比例的总体均值及 95% 置信区间（基于个体平均）
        means = []
        cis = []
        for p in proportions:
            vals = agent_avg_return[agent_avg_return['proportion'] == p]['returned_amount'].values
            mean_val = np.mean(vals)
            sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
            ci = sem * 1.96  # 近似 95% CI，也可用 t 分布
            means.append(mean_val)
            cis.append(ci)

        # 3. 效应量：各比例相对于参考比例的均值差（差值）及其 95% CI
        # 获取参考组的个体均值
        ref_vals = agent_avg_return[agent_avg_return['proportion'] == reference_proportion]['returned_amount'].values
        ref_mean = np.mean(ref_vals)
        ref_sem = np.std(ref_vals, ddof=1) / np.sqrt(len(ref_vals))

        effect_sizes = []
        effect_cis = []
        for p in proportions:
            if p == reference_proportion:
                effect_sizes.append(0.0)
                effect_cis.append(0.0)
            else:
                vals = agent_avg_return[agent_avg_return['proportion'] == p]['returned_amount'].values
                mean_diff = np.mean(vals) - ref_mean
                # 两组独立样本的差值的标准误
                sem_diff = np.sqrt((np.std(vals, ddof=1) ** 2 / len(vals)) +
                                   (np.std(ref_vals, ddof=1) ** 2 / len(ref_vals)))
                ci_diff = sem_diff * 1.96
                effect_sizes.append(mean_diff)
                effect_cis.append(ci_diff)

        # 4. 创建画布和子图
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 8),
                                                sharex=True,
                                                gridspec_kw={'height_ratios': [2, 1],
                                                             'hspace': 0.05})

        # 颜色定义
        olive_green = '#6B8E23'  # 橄榄绿
        reddish_brown = '#A0522D'  # 红棕色

        # ========== 上半部分：箱线图 + 均值点图 ==========
        # 箱线图
        box = ax_top.boxplot(data_for_box, positions=proportions, widths=0.6,
                             patch_artist=True,
                             boxprops=dict(facecolor=olive_green, color='black', linewidth=1.2),
                             whiskerprops=dict(color='black', linewidth=1.2),
                             capprops=dict(color='black', linewidth=1.2),
                             medianprops=dict(color='black', linewidth=1.5),
                             flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5))
        # 添加均值点图（红棕色）
        ax_top.errorbar(proportions, means, yerr=cis, fmt='o', color=reddish_brown,
                        capsize=5, capthick=1.5, markersize=8, label='Mean ± 95% CI')
        ax_top.set_ylabel('Returned amount', fontsize=12)
        ax_top.set_title('', fontsize=14)
        ax_top.legend(loc='upper right', frameon=False)
        ax_top.grid(axis='y', linestyle='--', alpha=0.5)

        # ========== 下半部分：效应量点图 ==========
        # 只显示非参考比例的点
        effect_props = [p for p in proportions if p != reference_proportion]
        effect_vals = [effect_sizes[i] for i, p in enumerate(proportions) if p != reference_proportion]
        effect_errs = [effect_cis[i] for i, p in enumerate(proportions) if p != reference_proportion]

        ax_bottom.errorbar(effect_props, effect_vals, yerr=effect_errs, fmt='o',
                           color=reddish_brown, capsize=5, capthick=1.5, markersize=8)
        # 添加参考线 y=0
        ax_bottom.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax_bottom.set_ylabel('Effect size\n(difference from 0% free)', fontsize=12)
        ax_bottom.set_xlabel('Fraction of free players', fontsize=12)
        ax_bottom.set_xticks(proportions)
        ax_bottom.set_xticklabels([f'{int(p * 100)}%' for p in proportions])
        ax_bottom.grid(axis='y', linestyle='--', alpha=0.5)

        # 设置纵轴范围（可选）
        all_return_vals = agent_avg_return['returned_amount'].values
        y_top_max = np.percentile(all_return_vals, 95) + 1
        y_top_min = max(0, np.percentile(all_return_vals, 5) - 1)
        ax_top.set_ylim(y_top_min, y_top_max)

        # 效应量纵轴自动调整
        all_effects = effect_vals
        if all_effects:
            y_bottom_max = max(all_effects) + max(effect_errs) * 1.2
            y_bottom_min = min(all_effects) - max(effect_errs) * 1.2
            ax_bottom.set_ylim(y_bottom_min, y_bottom_max)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"组合图已保存至: {output_path}")

    def plot_figures(self, output_dir: str, proportions: List[float] = None):
        """
        生成论文同款图表：
        - 图1: 委托/返还金额随轮次变化（不同比例对比）
        - 图2: 基尼系数条形图
        - 图3: 行为表型分布堆叠柱状图
        - 图4: 个人收益分布箱线图
        """
        if proportions is None:
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

        # 图1b：返还金额随轮次变化
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

        # 图2：基尼系数条形图
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

        # 图3：行为表型分布堆叠柱状图（百分比）
        phenotype_counts = []
        order = ['prosocial', 'neutral', 'antisocial']
        for prop in proportions:
            clustering = self.behavior_clustering(proportion=prop, n_clusters=3)
            counts = clustering['phenotype'].value_counts()
            cnts = [counts.get(p, 0) for p in order]
            phenotype_counts.append(cnts)
        phenotype_counts = np.array(phenotype_counts)
        phenotype_pct = phenotype_counts / phenotype_counts.sum(axis=1, keepdims=True) * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        bottom = np.zeros(len(proportions))
        colors_pheno = ['#2ecc71', '#f39c12', '#e74c3c']  # 绿、橙、红
        for i, (pheno, color) in enumerate(zip(order, colors_pheno)):
            ax.bar(prop_labels, phenotype_pct[:, i], bottom=bottom,
                   label=pheno, color=color, edgecolor='black')
            bottom += phenotype_pct[:, i]
        ax.set_xlabel("自由玩家比例")
        ax.set_ylabel("百分比 (%)")
        ax.set_title("行为表型分布")
        ax.legend(title="表型")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "phenotype_distribution.png"), dpi=150)
        plt.close()

        # 图4：个人收益分布箱线图
        plt.figure(figsize=(10, 6))
        data_to_plot = []
        for prop in proportions:
            metrics = self.compute_core_metrics(proportion=prop)
            agent_payoffs = metrics['agent_total_payoffs'].values
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

    # 默认 CSV 文件路径
    default_csv_path = os.path.join(
        CommonUtils.get_project_root_path(),
        "outputs",
        "20260422_112210_qwen-flash_trust_game",
        "20260422_112210_trust_game_p-[0, 0.25, 0.5, 0.75, 1].csv"
    )
    csv_path = default_csv_path

    if os.path.exists(csv_path):
        print(f"分析文件: {csv_path}")

        analyzer = TrustGameAnalyzer([csv_path])

        output_dir = os.path.dirname(csv_path)
        report_path = os.path.join(output_dir, "analysis_report.txt")
        figures_dir = os.path.join(output_dir, "figures")

        analyzer.generate_report(report_path)
        analyzer.plot_figures(figures_dir)

        combined_fig_path = os.path.join(figures_dir, "returned_amount_effect.png")
        analyzer.plot_returned_amount_with_effect_size(combined_fig_path)

        print("\n分析完成！")
        print(f"报告保存至: {report_path}")
        print(f"图表保存至: {figures_dir}")
    else:
        print(f"文件不存在: {csv_path}")
        print(f"使用方法: python analysy.py <csv_file_path>")
        print(f"示例: python analysy.py outputs/xxx.csv")