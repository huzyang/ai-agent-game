"""
自由玩家：对每个邻居独立决策。
"""

from typing import Dict, Any
from src.experiment.base_agent import BaseAgent
import random


class FreeAgent(BaseAgent):
    def decide(self, game_name: str, context: Dict[str, Any]) -> Dict[int, Any]:
        """
        对每个邻居分别调用LLM决策。
        """
        decisions = {}
        for neighbor in self.neighbors:
            # 构建个性化上下文（包含该邻居的历史）
            neighbor_context = self._get_neighbor_context(neighbor, context)
            system_prompt = "你是一个参与实验的人类被试。请仅输出决策结果，不要输出其他内容。"
            user_prompt = self._build_prompt(game_name, neighbor_context, neighbor)
            raw = self._call_llm(system_prompt, user_prompt)
            decisions[neighbor] = self._parse_decision(game_name, raw)
        return decisions

    def _get_neighbor_context(self, neighbor: int, global_context: Dict[str, Any]) -> Dict[str, Any]:
        """提取针对特定邻居的历史信息"""
        last = self.get_last_round()
        if last:
            # 假设全局context中包含各邻居的历史
            neighbor_last_choice = global_context.get("neighbor_last_choices", {}).get(neighbor)
            neighbor_last_return = global_context.get("neighbor_last_returns", {}).get(neighbor)
            last_accepted = global_context.get("last_accepted", {}).get(neighbor)
            return {
                "opponent_last_choice": neighbor_last_choice,
                "opponent_last_return": neighbor_last_return,
                "last_accepted": last_accepted,
                "last_my_decision": last["decisions"].get(neighbor),
                "last_my_payoff": last["payoffs"].get(neighbor)
            }
        return {}

    def _build_prompt(self, game_name: str, context: Dict[str, Any], neighbor: int) -> str:
        """构建个性化提示"""
        if game_name == "PrisonerDilemma":
            hist_str = ""
            if context.get("last_my_decision") is not None:
                hist_str = f"上一轮你选择了 {context['last_my_decision']}，对方选择了 {context.get('opponent_last_choice', '?')}，你的收益为 {context.get('last_my_payoff', '?')}。\n"
            return (
                f"你正在与一个邻居重复进行囚徒困境博弈。\n"
                f"收益规则：双方合作各得3，一方合作一方背叛得(0,5)，双方背叛各得1。\n"
                f"{hist_str}"
                f"本轮请选择合作(C)或背叛(D)："
            )
        elif game_name == "TrustGame":
            hist_str = ""
            if context.get("last_my_decision") is not None:
                hist_str = f"上一轮你委托了{context['last_my_decision']}，对方返还了{context.get('opponent_last_return', '?')}，你的收益为{context.get('last_my_payoff', '?')}。\n"
            return (
                f"你正在与一个邻居重复进行信任博弈。\n"
                f"每轮你先决定委托金额（0-10），对方收到3倍后返还部分。\n"
                f"{hist_str}"
                f"本轮你愿意委托多少？请输出0到10之间的整数："
            )
        elif game_name == "UltimatumGame":
            hist_str = ""
            if context.get("last_my_decision") is not None:
                hist_str = f"上一轮你提议{context['last_my_decision']}，对方{'接受' if context.get('last_accepted') else '拒绝'}，你的收益为{context.get('last_my_payoff', '?')}。\n"
            return (
                f"你正在与一个邻居重复进行最后通牒博弈。\n"
                f"每轮你提议如何分配10个代币（自己留10-提议，给对方提议）。\n"
                f"{hist_str}"
                f"本轮你提议给对方多少？请输出0到10之间的整数："
            )
        else:
            raise ValueError(f"Unknown game: {game_name}")

    def _parse_decision(self, game_name: str, raw: str) -> Any:
        raw = raw.strip().upper()
        if game_name == "PrisonerDilemma":
            if raw in ("C", "COOPERATE"):
                return "C"
            elif raw in ("D", "DEFECT"):
                return "D"
            else:
                print(f"无法解析决策'{raw}'，随机选择")
                return random.choice(["C", "D"])
        else:
            try:
                val = int(raw)
                if 0 <= val <= 10:
                    return val
                else:
                    return 5
            except:
                return 5