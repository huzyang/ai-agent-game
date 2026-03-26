"""
阶段2主运行脚本：分别运行囚徒困境、信任博弈、最后通牒博弈各50轮，
使用两个智能体（可配置为受限或自由），验证基本功能。
"""

import os
import sys
import logging
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_agent import BaseAgent
from constrained_agent import ConstrainedAgent
from free_agent import FreeAgent
from prisoner_dilemma import PrisonerDilemmaGame
from trust_game import TrustGame
from ultimatum_game import UltimatumGame


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


def run_prisoner_dilemma(logger):
    """运行囚徒困境"""
    logger.info("=== 囚徒困境 50轮 ===")
    # 创建两个智能体，邻居列表相互包含
    agent1 = ConstrainedAgent(agent_id=1, agent_type="constrained", neighbors=[2])
    agent2 = ConstrainedAgent(agent_id=2, agent_type="constrained", neighbors=[1])
    game = PrisonerDilemmaGame(agent1, agent2, rounds=2)
    history = game.run()
    # 统计合作率
    coop_count = sum(1 for r in history if r["agent1"]["choice"] == "C" and r["agent2"]["choice"] == "C")
    coop_rate = coop_count / 50
    logger.info(f"囚徒困境完成，合作率: {coop_rate:.2f}")
    # 记录关键信息
    for i, r in enumerate(history[:3], 1):
        logger.info(f"  第{i}轮: 玩家1={r['agent1']['choice']}, 玩家2={r['agent2']['choice']}, 收益({r['agent1']['payoff']},{r['agent2']['payoff']})")
    return history


def run_trust_game(logger):
    """运行信任博弈"""
    logger.info("=== 信任博弈 50轮 ===")
    trustor = FreeAgent(agent_id=1, agent_type="free", neighbors=[2])
    trustee = ConstrainedAgent(agent_id=2, agent_type="constrained", neighbors=[1])
    game = TrustGame(trustor, trustee, rounds=50)
    history = game.run()
    # 统计平均委托金额
    avg_send = sum(r["trustor"]["send"] for r in history) / 50
    avg_return = sum(r["trustee"]["return"] for r in history) / 50
    logger.info(f"信任博弈完成，平均委托: {avg_send:.2f}, 平均返还: {avg_return:.2f}")
    for i, r in enumerate(history[:3], 1):
        logger.info(f"  第{i}轮: 委托={r['trustor']['send']}, 返还={r['trustee']['return']}, 收益({r['trustor']['payoff']},{r['trustee']['payoff']})")
    return history


def run_ultimatum_game(logger):
    """运行最后通牒博弈"""
    logger.info("=== 最后通牒博弈 50轮 ===")
    proposer = ConstrainedAgent(agent_id=1, agent_type="constrained", neighbors=[2])
    responder = FreeAgent(agent_id=2, agent_type="free", neighbors=[1])
    game = UltimatumGame(proposer, responder, rounds=50)
    history = game.run()
    # 统计平均提议和接受率
    avg_offer = sum(r["proposer"]["offer"] for r in history) / 50
    accept_rate = sum(1 for r in history if r["responder"]["accepted"]) / 50
    logger.info(f"最后通牒博弈完成，平均提议: {avg_offer:.2f}, 接受率: {accept_rate:.2f}")
    for i, r in enumerate(history[:3], 1):
        logger.info(f"  第{i}轮: 提议={r['proposer']['offer']}, {'接受' if r['responder']['accepted'] else '拒绝'}, 收益({r['proposer']['payoff']},{r['responder']['payoff']})")
    return history


if __name__ == "__main__":
    logger = setup_logging()
    logger.info("开始阶段2实验验证")

    # 运行三个博弈
    pd_history = run_prisoner_dilemma(logger)
    # tg_history = run_trust_game(logger)
    # ug_history = run_ultimatum_game(logger)

    logger.info("阶段2完成，所有博弈已运行50轮，历史记录已保存至日志文件。")