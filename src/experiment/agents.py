import random
import mesa
import numpy as np
from enum import IntEnum
from camel.agents import ChatAgent
from mesa.discrete_space import CellAgent


class BaseAgent(CellAgent):
    """Agent member of the iterated, spatial prisoner's dilemma model."""

    def __init__(self, model, unique_id, cell=None):
        """
        Create a new Base agent.
        """
        super().__init__(model)
        self.unique_id = unique_id
        self.cell = cell

        self.invested_amounts = []
        self.received_amounts = []
        self.payoff = 0.0  # 每一轮的最终收益

        self.llm_agent = None

    def __str__(self):
        return f"Agent {self.unique_id}: neighbors: {self.neighbors}; Payoff: {self.payoff:.2f}"

    def set_llm_agent(self, llm_agent: ChatAgent) -> None:
        self.llm_agent = llm_agent
    @property
    def neighbor_ids(self):
        # 邻居id列表
        return sorted([agent.unique_id for agent in self.cell.neighborhood.agents])
    def update_payoff(self):
        pass

    def step(self):
        pass
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
        pass

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用LLM获取原始响应"""
        pass
