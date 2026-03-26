"""
囚徒困境博弈类：处理两个玩家的多轮交互。
"""

from typing import List, Tuple, Dict, Any
from .base_agent import BaseAgent

PAYOFF_MATRIX = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}


class PrisonerDilemmaGame:
    def __init__(self, agent1: BaseAgent, agent2: BaseAgent, rounds: int = 50):
        self.agents = [agent1, agent2]
        self.rounds = rounds
        self.history = []  # 每轮结果存储

    def run(self) -> List[Dict]:
        """运行多轮，返回历史"""
        for r in range(1, self.rounds + 1):
            # 构建上下文：各agent上一轮对方的决策
            context = self._build_context()
            # 获取决策
            decisions1 = self.agents[0].decide("PrisonerDilemma", context.get(0, {}))
            decisions2 = self.agents[1].decide("PrisonerDilemma", context.get(1, {}))
            # 由于只有两个玩家，我们取决策（注意邻居列表可能含对方ID）
            a1_choice = decisions1.get(self.agents[1].id, "D")
            a2_choice = decisions2.get(self.agents[0].id, "D")
            # 计算收益
            payoff1, payoff2 = PAYOFF_MATRIX[(a1_choice, a2_choice)]
            # 记录
            self.agents[0].add_history(r, {self.agents[1].id: a1_choice}, {self.agents[1].id: payoff1})
            self.agents[1].add_history(r, {self.agents[0].id: a2_choice}, {self.agents[0].id: payoff2})
            self.history.append({
                "round": r,
                "agent1": {"id": self.agents[0].id, "choice": a1_choice, "payoff": payoff1},
                "agent2": {"id": self.agents[1].id, "choice": a2_choice, "payoff": payoff2}
            })
        return self.history

    def _build_context(self) -> Dict[int, Dict]:
        """构建每个agent的历史上下文"""
        context = {0: {}, 1: {}}
        last_round = self.history[-1] if self.history else None
        if last_round:
            context[0]["opponent_last_choice"] = last_round["agent2"]["choice"]
            context[1]["opponent_last_choice"] = last_round["agent1"]["choice"]
        return context