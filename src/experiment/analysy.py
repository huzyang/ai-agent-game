"""
analysis.py
信任博弈实验数据分析模块
功能：计算核心指标、行为聚类、统计检验、生成图表
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from typing import List, Dict, Tuple, Optional
import logging
from src.utils import CommonUtils
logger = logging.getLogger(__name__)
# 设置中文字体（Windows系统）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class TrustGameAnalyzer:
    """
    信任博弈实验分析器
    支持加载多个CSV文件（不同proportion或重复实验），计算论文核心指标，
    进行行为表型聚类和统计检验，并生成可视化图表。
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
                round, proportion, investor_agent_id, investor_agent_type,
                trustee_neighbor_id, invested_amount, returned_amount,
                investor_agent_payoff, trustee_agent_payoff
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
        # 确保数据类型
        self.df['invested_amount'] = pd.to_numeric(self.df['invested_amount'], errors='coerce').fillna(0)
        self.df['returned_amount'] = pd.to_numeric(self.df['returned_amount'], errors='coerce').fillna(0)
        self.df['investor_agent_payoff'] = pd.to_numeric(self.df['investor_agent_payoff'], errors='coerce').fillna(0)
        self.df['trustee_agent_payoff'] = pd.to_numeric(self.df['trustee_agent_payoff'], errors='coerce').fillna(0)

        # 添加辅助列：每笔交互的总收益（投资者+受托者）
        self.df['interaction_total_payoff'] = self.df['investor_agent_payoff'] + self.df['trustee_agent_payoff']

        logger.info(f"数据合并完成，总行数: {len(self.df)}")

        # 设置绘图风格
        sns.set_theme(style="whitegrid")
        plt.rcParams.update(self.STYLE)

    @staticmethod
    def gini_coefficient(values: np.ndarray) -> float:
        """计算基尼系数"""
        values = values[values >= 0]  # 过滤负值
        if len(values) == 0 or np.sum(values) == 0:
            return 0.0
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_vals)) - (n+1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals))
        return gini

    def compute_core_metrics(self, proportion: Optional[float] = None) -> Dict:
        """
        计算核心指标，可指定proportion筛选
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
            raise ValueError(f"未找到proportion={proportion}的数据")

        # 1. 平均委托/返还金额（按交互记录）
        mean_invested = df['invested_amount'].mean()
        mean_returned = df['returned_amount'].mean()

        # 2. 群体总收益（所有交互的收益之和）
        total_payoff_sum = df['interaction_total_payoff'].sum()

        # 3. 个人累计收益（按agent_id聚合）
        # 每个agent作为投资者和受托者时收益已分别记录，这里需要聚合每个agent的总收益
        # 注意：每个agent作为投资者时的收益在 investor_agent_payoff 字段，作为受托者时的收益在 trustee_agent_payoff 字段
        # 但我们的数据中，每行是一个交互，投资者收益记录在 investor_agent_payoff，受托者收益记录在 trustee_agent_payoff
        # 因此，对于agent A，其总收益 = 所有A作为投资者时的 investor_agent_payoff + 所有A作为受托者时的 trustee_agent_payoff
        # 需要分别聚合
        investor_sum = df.groupby('investor_agent_id')['investor_agent_payoff'].sum()
        trustee_sum = df.groupby('trustee_neighbor_id')['trustee_agent_payoff'].sum()
        agent_total = investor_sum.add(trustee_sum, fill_value=0).sort_index()
        gini = self.gini_coefficient(agent_total.values)

        # 4. 按轮次聚合
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
        分类依据：聚类后按平均委托金额排序，高->亲社会，中->中性，低->反社会
        Returns:
            DataFrame包含每个agent_id及其聚类标签、特征值
        """
        df = self.df if proportion is None else self.df[self.df['proportion'] == proportion]
        if len(df) == 0:
            raise ValueError(f"未找到proportion={proportion}的数据")

        # 提取每个agent的特征：平均委托金额、平均返还金额
        # 注意：每个agent可能作为投资者多次，但这里只考虑作为投资者的行为（因为返还金额也是针对其委托的邻居）
        # 但为了更全面，也可以同时考虑作为受托者时的返还行为。论文中聚类使用六变量，简化版使用两个核心指标。
        agent_investor = df.groupby('investor_agent_id').agg(
            mean_invested=('invested_amount', 'mean'),
            mean_returned_from_trustee=('returned_amount', 'mean')
        ).reset_index()

        # 标准化
        features = agent_investor[['mean_invested', 'mean_returned_from_trustee']].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)

        agent_investor['cluster'] = labels

        # 根据平均委托金额排序确定标签含义
        cluster_order = agent_investor.groupby('cluster')['mean_invested'].mean().sort_values().index.tolist()
        # 最小平均委托 -> antisocial, 中间 -> neutral, 最大 -> prosocial
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
        对两组不同proportion的实验进行独立样本t检验
        Args:
            group1_proportion: 第一组的自由玩家比例
            group2_proportion: 第二组的自由玩家比例
            metric: 比较的指标，可选 'invested_amount', 'returned_amount', 'investor_agent_payoff'
        Returns:
            包含t统计量和p值的字典
        """
        df1 = self.df[self.df['proportion'] == group1_proportion]
        df2 = self.df[self.df['proportion'] == group2_proportion]

        if len(df1) == 0 or len(df2) == 0:
            raise ValueError("指定的比例在数据中不存在")

        # 按agent聚合，避免个体内相关性（也可按每笔交互，但t检验假设独立性）
        if metric == 'invested_amount':
            # 每个agent的平均委托金额
            vals1 = df1.groupby('investor_agent_id')['invested_amount'].mean().dropna()
            vals2 = df2.groupby('investor_agent_id')['invested_amount'].mean().dropna()
        elif metric == 'returned_amount':
            vals1 = df1.groupby('investor_agent_id')['returned_amount'].mean().dropna()
            vals2 = df2.groupby('investor_agent_id')['returned_amount'].mean().dropna()
        elif metric == 'investor_agent_payoff':
            # 每个agent的总投资者收益
            vals1 = df1.groupby('investor_agent_id')['investor_agent_payoff'].sum().dropna()
            vals2 = df2.groupby('investor_agent_id')['investor_agent_payoff'].sum().dropna()
        else:
            raise ValueError("metric 必须是 'invested_amount', 'returned_amount' 或 'investor_agent_payoff'")

        t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False)  # Welch's t-test
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
            proportions: 要分析的proportion列表，默认为数据中所有的唯一比例
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

            # 统计检验（如果有至少两个比例）
            if len(proportions) >= 2:
                f.write("【统计检验结果】\n")
                # 比较全受限(0)和全自由(1)（如果存在）
                if 0.0 in proportions and 1.0 in proportions:
                    res = self.t_test_analysis(0.0, 1.0, metric='invested_amount')
                    f.write(f"  委托金额 t检验 (0%自由 vs 100%自由): t={res['t_statistic']:.3f}, p={res['p_value']:.4f}\n")
                    res2 = self.t_test_analysis(0.0, 1.0, metric='returned_amount')
                    f.write(f"  返还金额 t检验 (0%自由 vs 100%自由): t={res2['t_statistic']:.3f}, p={res2['p_value']:.4f}\n")

            # 行为表型聚类（每个比例单独聚类）
            f.write("\n【行为表型聚类结果】\n")
            for prop in proportions:
                clustering = self.behavior_clustering(proportion=prop, n_clusters=3)
                counts = clustering['phenotype'].value_counts()
                f.write(f"  比例 {prop:.0%}:\n")
                for pheno, cnt in counts.items():
                    f.write(f"    {pheno}: {cnt} 人 ({cnt/len(clustering)*100:.1f}%)\n")
                f.write("\n")

        logger.info(f"分析报告已保存至: {output_path}")

    def plot_figures(self, output_dir: str, proportions: List[float] = None):
        """
        生成论文同款图表：
        - 图1: 委托/返还金额随轮次变化（不同比例对比）
        - 图2: 基尼系数条形图（不同比例）
        - 图3: 行为表型分布饼图/堆叠柱状图
        - 图4: 个人收益分布箱线图
        """
        if proportions is None:
            proportions = sorted(self.df['proportion'].unique())

        os.makedirs(output_dir, exist_ok=True)

        # 设置颜色方案
        colors = sns.color_palette("Blues_d", n_colors=len(proportions))
        prop_labels = [f"{int(p*100)}%" for p in proportions]

        # 图1：委托金额随轮次变化（折线图）
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
        plt.show()
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
        plt.show()
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
        plt.ylim(0, max(ginis)*1.2 if max(ginis)>0 else 0.6)
        # 添加数值标签
        for bar, g in zip(bars, ginis):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{g:.3f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gini_coefficient.png"), dpi=150)
        plt.show()
        plt.close()

        # 图3：行为表型分布堆叠柱状图
        phenotype_counts = []
        for prop in proportions:
            clustering = self.behavior_clustering(proportion=prop, n_clusters=3)
            counts = clustering['phenotype'].value_counts()
            # 确保顺序 prosocial, neutral, antisocial
            order = ['prosocial', 'neutral', 'antisocial']
            cnts = [counts.get(p, 0) for p in order]
            phenotype_counts.append(cnts)
        phenotype_counts = np.array(phenotype_counts)
        # 转换为百分比
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
        plt.show()
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
        plt.show()
        plt.close()

        logger.info(f"所有图表已保存至: {output_dir}")


# 命令行使用示例
if __name__ == "__main__":
    # 示例：分析某个输出目录下的所有CSV文件
    import glob
    csv_dir = "outputs/20260416_170000_qwen3.5-flash_trust_game/trust_game_p-0.5_1.csv"
    csv_files = glob.glob(os.path.join(CommonUtils.get_project_root_path(),csv_dir))
    if csv_files:
        analyzer = TrustGameAnalyzer(csv_files)
        analyzer.generate_report("analysis_report.txt")
        analyzer.plot_figures("figures")
        print("分析完成！")