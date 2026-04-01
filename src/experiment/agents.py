import random
import mesa
import numpy as np
from enum import IntEnum
from camel.agents import ChatAgent

class StrategyType(IntEnum):
    """离散策略类型枚举 - 使用整数表示"""
    C = "C"  # 投资 + 可信
    D = "D"  # 投资 + 不可信

    @classmethod
    def from_string(cls, strategy_str):
        """从字符串转换"""
        strategy_map = {
            "C": cls.C,
            "D": cls.D,
        }
        return strategy_map.get(strategy_str)
from mesa.discrete_space import CellAgent


class BaseAgent(CellAgent):
    """Agent member of the iterated, spatial prisoner's dilemma model."""

    def __init__(self, model, unique_id, cell=None):
        """
        Create a new Prisoner's Dilemma agent.

        Args:
            model: model instance
            starting_move: If provided, determines the agent's initial state:
                           C(ooperating) or D(efecting). Otherwise, random.
        """
        super().__init__(model)
        self.unique_id = unique_id
        self.cell = cell

        self.strategy = random.choice(list(StrategyType))
        self.strategy_history = []  # 策略历史
        self.payoff = 0.0  # 每一轮的最终收益
        self.neighbors = []  # 邻居id

        self.llm_client = ChatAgent(
            system_message="你是一个好奇的智能体，正在探索宇宙的奥秘。",
            model=self.model.llm_model,
            output_language='Chinese'
        )


    @property
    def is_cooperating(self):
        return self.strategy == "C"

    @property
    def is_llm_client(self):
        return False if self.llm_client is None else True

    def __str__(self):
        return f"Agent {self.unique_id}: LLM Agent: {self.is_llm_client}; Strategy: (now: {self.strategy}, next: {self.next_strategy}); Payoff: {self.payoff:.2f}"

    def decide(self):
        """
        根据游戏类型和上下文做出决策。
        """
        # 1、构建提示词

        # 2、调用LLM获取决策

        # 3、解析决策
        pass

    def update_payoff(self):
        # neighbors = [*list(self.cell.neighborhood.agents), self]
        neighbors = self.cell.neighborhood.agents
        strategies = [neighbor.strategy for neighbor in neighbors]
        return sum(self.model.payoff[(self.strategy, strategy)] for strategy in strategies)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用LLM获取原始响应"""
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=50,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Agent {self.id} LLM调用失败: {e}")
            return ""
