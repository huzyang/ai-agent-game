#!/usr/bin/env python
# coding: utf-8

"""
信任博弈数据分析脚本
- 分别处理 sent_amount 和 returned_amount
- 去重规则：同一 round, proportion, trustor_id, 金额值 只保留一条
- 计算各 proportion 分组的箱线图统计量
- 生成散点 jitter 数据
- 保存到 plot_data.xlsx 的两个 sheet 中
- 绘制 sent_amount 的复刻图（散点+箱线图，无异常值）
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.utils import CommonUtils

# 设置随机种子和字体
np.random.seed(42)
mpl.rcParams['font.sans-serif'] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
mpl.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 读取原始数据
# ============================================================
# 默认 CSV 文件路径
csv_dir = os.path.join(
    CommonUtils.get_project_root_path(),
    "outputs",
    "deepseek-v4-flash_trust_game"
)
file_name = 'deepseek-v4-flash_trust_game_plot_data_p-[0, 0.25,0.5,0.75,1].xlsx'
file_path = os.path.join(csv_dir, file_name)
df_raw = pd.read_excel(file_path, sheet_name='sent_returned_amount')
print("原始数据形状:", df_raw.shape)
print("列名:", df_raw.columns.tolist())


# ============================================================
# 2. 定义处理函数：去重 + 统计 + 散点jitter
# ============================================================
def process_amount_column(df, amount_col):
    """
    对某一金额列（sent_amount 或 returned_amount）进行去重和统计
    返回:
        - stats_df: 每个proportion的统计量 (DataFrame)
        - scatter_df: 散点数据，包含 proportion, amount, x_jitter
    """
    # 去重：同一 round, proportion, trustor_id, 金额 只保留第一个
    cols_dedupe = ['round', 'proportion', 'trustor_id', amount_col]
    df_dedup = df.drop_duplicates(subset=cols_dedupe, keep='first').copy()

    # 移除缺失值
    df_dedup = df_dedup.dropna(subset=[amount_col])

    # 计算每个 proportion 分组的统计量
    stats_list = []
    for prop, group in df_dedup.groupby('proportion'):
        vals = group[amount_col].values
        q1 = np.percentile(vals, 25)
        median = np.median(vals)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1
        lower_whisker = np.min(vals[vals >= q1 - 1.5 * iqr]) if len(vals[vals >= q1 - 1.5 * iqr]) > 0 else q1 - 1.5 * iqr
        upper_whisker = np.max(vals[vals <= q3 + 1.5 * iqr]) if len(vals[vals <= q3 + 1.5 * iqr]) > 0 else q3 + 1.5 * iqr
        stats_list.append({
            'proportion': prop,
            'count': len(vals),
            'min': np.min(vals),
            'q1': q1,
            'median': median,
            'q3': q3,
            'max': np.max(vals),
            'lower_whisker': lower_whisker,
            'upper_whisker': upper_whisker,
            'mean': np.mean(vals),
            'std': np.std(vals)
        })
    stats_df = pd.DataFrame(stats_list).sort_values('proportion')

    # 生成散点数据（添加 jitter）
    jitter_strength = 0.02
    scatter_records = []
    for prop, group in df_dedup.groupby('proportion'):
        vals = group[amount_col].values
        jitters = np.random.uniform(-jitter_strength, jitter_strength, size=len(vals))
        x_jitter = prop + jitters
        for v, xj in zip(vals, x_jitter):
            scatter_records.append({
                'proportion': prop,
                'amount': v,
                'x_jitter': xj
            })
    scatter_df = pd.DataFrame(scatter_records)

    return stats_df, scatter_df


# 处理 sent_amount
sent_stats, sent_scatter = process_amount_column(df_raw, 'sent_amount')
# 处理 returned_amount
ret_stats, ret_scatter = process_amount_column(df_raw, 'returned_amount')

# ============================================================
# 3. 保存到 Excel 的两个工作表
# ============================================================
output_path = os.path.join(csv_dir, 'plot_data.xlsx')
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # sent_amount 工作表：左边统计，右边散点（用不同列区域，不加合并，以便读取）
    # 为清晰起见，将统计量和散点数据分开放置，中间空一列
    sent_stats.to_excel(writer, sheet_name='sent_amount', startrow=0, startcol=0, index=False)
    sent_scatter.to_excel(writer, sheet_name='sent_amount', startrow=0, startcol=len(sent_stats.columns) + 2, index=False)

    # returned_amount 工作表
    ret_stats.to_excel(writer, sheet_name='returned_amount', startrow=0, startcol=0, index=False)
    ret_scatter.to_excel(writer, sheet_name='returned_amount', startrow=0, startcol=len(ret_stats.columns) + 2, index=False)

print(f"处理后的数据已保存至: {output_path}")

# ============================================================
# 4. 绘制 sent_amount 的复刻图（散点 + 箱线图，无异常值）
# ============================================================
plt.figure(figsize=(8, 6))

# 获取所有 proportion 以及对应的 sent_amount 原始值（去重后的）
proportions = sorted(sent_scatter['proportion'].unique())
box_data = []
for p in proportions:
    vals = sent_scatter[sent_scatter['proportion'] == p]['amount'].values
    box_data.append(vals)

# 绘制箱线图（无异常值点）
bp = plt.boxplot(box_data, positions=proportions, widths=0.15,
                 patch_artist=True, showfliers=False,
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'))

# 绘制带 jitter 的散点图
for p in proportions:
    subset = sent_scatter[sent_scatter['proportion'] == p]
    plt.scatter(subset['x_jitter'], subset['amount'],
                s=15, alpha=0.5, c='dimgray', edgecolors='none')

# 坐标轴设置
plt.xlabel('Fraction of free players', fontsize=12)
plt.ylabel('Sent amount', fontsize=12)
plt.xticks(proportions, [f'{p:.2f}' for p in proportions], fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(csv_dir, 'sent_amount_plot.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Sent amount 图已保存为: sent_amount_plot.png")

# 如果需要同样绘制 returned_amount 的图，可取消下面注释
# plt.figure(figsize=(8,6))
# ... 类似代码 ...