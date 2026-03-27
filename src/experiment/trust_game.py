"""
信任博弈：两阶段，支持多轮，记录历史。
"""

from typing import List, Dict, Any
from base_agent import BaseAgent


class TrustGame:
    def __init__(self, trustor: BaseAgent, trustee: BaseAgent, rounds: int = 50):
        self.trustor = trustor
        self.trustee = trustee
        self.rounds = rounds
        self.history = []  # 存储每轮结果

    def run(self) -> List[Dict]:
        """运行多轮信任博弈"""
        for r in range(1, self.rounds + 1):
            # 构建上下文：上一轮的委托和返还
            context = self._build_context()

            # 第一阶段：委托
            # trustor 做出委托决策（返回 {trustee_id: amount}）
            decisions = self.trustor.decide("TrustGame", context.get(self.trustor.id, {}))
            send_amount = decisions.get(self.trustee.id, 5)  # 默认5

            # 第二阶段：返还
            # 受托方收到 send_amount * 3，决定返还多少
            # 构建受托方上下文（包含收到金额）
            trustee_context = {
                "received": send_amount * 3,
                "last_return": context.get(self.trustee.id, {}).get("opponent_last_return", None),
                "last_payoff": context.get(self.trustee.id, {}).get("last_my_payoff", None),
                "last_my_decision": context.get(self.trustee.id, {}).get("last_my_decision", None)
            }
            # 受托方决策（返回 {trustor_id: return_amount}）
            trustee_decisions = self.trustee.decide("TrustGame", trustee_context)
            return_amount = trustee_decisions.get(self.trustor.id, 0)

            # 计算收益
            trustor_payoff = send_amount * 3 - return_amount
            trustee_payoff = return_amount

            # 记录历史
            self.trustor.add_history(r, {self.trustee.id: send_amount}, {self.trustee.id: trustor_payoff})
            self.trustee.add_history(r, {self.trustor.id: return_amount}, {self.trustor.id: trustee_payoff})

            self.history.append({
                "round": r,
                "trustor": {"id": self.trustor.id, "send": send_amount, "payoff": trustor_payoff},
                "trustee": {"id": self.trustee.id, "return": return_amount, "payoff": trustee_payoff}
            })
        return self.history

    def _build_context(self) -> Dict[int, Dict]:
        """构建每个agent的上下文（上一轮对方的行为）"""
        context = {self.trustor.id: {}, self.trustee.id: {}}
        if self.history:
            last = self.history[-1]
            context[self.trustor.id]["opponent_last_return"] = last["trustee"]["return"]
            context[self.trustee.id]["opponent_last_return"] = last["trustor"]["send"]
        return context