"""
最后通牒博弈：两阶段，支持多轮，记录历史。
"""

from typing import List, Dict, Any
from .base_agent import BaseAgent


class UltimatumGame:
    def __init__(self, proposer: BaseAgent, responder: BaseAgent, rounds: int = 50):
        self.proposer = proposer
        self.responder = responder
        self.rounds = rounds
        self.history = []

    def run(self) -> List[Dict]:
        """运行多轮最后通牒博弈"""
        for r in range(1, self.rounds + 1):
            context = self._build_context()

            # 第一阶段：提议
            decisions = self.proposer.decide("UltimatumGame", context.get(self.proposer.id, {}))
            offer = decisions.get(self.responder.id, 5)  # 默认提议5

            # 第二阶段：接受或拒绝
            # 构建响应方上下文（包含收到的提议）
            responder_context = {
                "offer": offer,
                "last_accepted": context.get(self.responder.id, {}).get("last_accepted", None),
                "last_payoff": context.get(self.responder.id, {}).get("last_my_payoff", None),
                "last_my_decision": context.get(self.responder.id, {}).get("last_my_decision", None)
            }
            # 响应方决策（接受/拒绝）
            # 注意：我们要求响应方输出 "ACCEPT" 或 "REJECT"
            responder_decisions = self.responder.decide("UltimatumGame", responder_context)
            accepted = responder_decisions.get(self.proposer.id, "REJECT") == "ACCEPT"

            # 计算收益
            if accepted:
                proposer_payoff = 10 - offer
                responder_payoff = offer
            else:
                proposer_payoff = 0
                responder_payoff = 0

            # 记录历史
            self.proposer.add_history(r, {self.responder.id: offer}, {self.responder.id: proposer_payoff})
            self.responder.add_history(r, {self.proposer.id: "ACCEPT" if accepted else "REJECT"},
                                       {self.proposer.id: responder_payoff})

            self.history.append({
                "round": r,
                "proposer": {"id": self.proposer.id, "offer": offer, "payoff": proposer_payoff},
                "responder": {"id": self.responder.id, "accepted": accepted, "payoff": responder_payoff}
            })
        return self.history

    def _build_context(self) -> Dict[int, Dict]:
        """构建每个agent的上下文（上一轮对方的行为）"""
        context = {self.proposer.id: {}, self.responder.id: {}}
        if self.history:
            last = self.history[-1]
            context[self.proposer.id]["last_accepted"] = last["responder"]["accepted"]
            context[self.responder.id]["last_accepted"] = last["responder"]["accepted"]
        return context