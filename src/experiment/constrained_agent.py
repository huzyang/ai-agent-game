"""
受限玩家：对所有邻居采取统一决策。
"""
import random
from typing import Dict, Any
from .base_agent import BaseAgent


class ConstrainedAgent(BaseAgent):
    def decide(self, game_name: str, context: Dict[str, Any]) -> Dict[int, Any]:
        """
        决策逻辑：生成一个统一决策，应用到所有邻居。
        context 包含游戏特定信息，如收益矩阵、历史交互等。
        """
        # 构建提示，让LLM输出统一决策
        system_prompt = "你是一个参与实验的人类被试。请仅输出决策结果，不要输出其他内容。"
        user_prompt = self._build_prompt(game_name, context)

        raw = self._call_llm(system_prompt, user_prompt)
        decision = self._parse_decision(game_name, raw)

        # 应用到所有邻居
        return {neighbor: decision for neighbor in self.neighbors}

    def _build_prompt(self, game_name: str, context: Dict[str, Any]) -> str:
        """根据游戏构建提示"""
        if game_name == "PrisonerDilemma":
            # 历史信息
            last = self.get_last_round()
            hist_str = ""
            if last:
                hist_str = f"上一轮你选择了 {last['decisions'][self.neighbors[0]]}，对方选择了 {context['opponent_last_choice']}，你的收益为 {last['payoffs'][self.neighbors[0]]}。\n"
            return (
                f"你正在参与一个重复囚徒困境博弈。\n"
                f"收益规则：双方合作各得3，一方合作一方背叛得(0,5)，双方背叛各得1。\n"
                f"{hist_str}"
                f"本轮请选择合作(C)或背叛(D)："
            )
        elif game_name == "TrustGame":
            last = self.get_last_round()
            hist_str = ""
            if last:
                hist_str = f"上一轮你委托了{last['decisions'][self.neighbors[0]]}，对方返还了{context['opponent_last_return']}，你的收益为{last['payoffs'][self.neighbors[0]]}。\n"
            return (
                f"你正在参与一个重复信任博弈。\n"
                f"每轮你先决定委托金额（0-10），对方收到3倍后返还部分。\n"
                f"{hist_str}"
                f"本轮你愿意委托多少？请输出0到10之间的整数："
            )
        elif game_name == "UltimatumGame":
            last = self.get_last_round()
            hist_str = ""
            if last:
                hist_str = f"上一轮你提议{last['decisions'][self.neighbors[0]]}，对方{'接受' if context['last_accepted'] else '拒绝'}，你的收益为{last['payoffs'][self.neighbors[0]]}。\n"
            return (
                f"你正在参与一个重复最后通牒博弈。\n"
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
        else:  # Trust/Ultimatum: 解析整数
            try:
                val = int(raw)
                if 0 <= val <= 10:
                    return val
                else:
                    return 5  # 默认中间值
            except:
                return 5