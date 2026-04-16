import random
import mesa
import numpy as np
from enum import IntEnum
from camel.agents import ChatAgent
from mesa.discrete_space import CellAgent


class BaseAgent(CellAgent):
    """Agent member of the iterated, spatial prisoner's dilemma model."""

    def __init__(self, model, cell=None):
        """
        Create a new Base agent.
        """
        super().__init__(model)
        # self.unique_id = unique_id
        self.cell = cell

        self.I_invested_1 = []
        self.T_received_2 = []
        self.T_returned_3 = []
        self.I_received_4 = []
        self.payoff = 0.0  # 每一轮的最终收益
        self.type_restriction = ""
        self.llm_agent = None

    def __str__(self):
        return f"Agent {self.unique_id}: Position: {self.cell.coordinate}; Neighbors: {self.neighbor_ids}; Payoff: {self.payoff:.2f}"

    def set_llm_agent(self, llm_agent: ChatAgent) -> None:
        self.llm_agent = llm_agent
    @property
    def neighbor_ids(self):
        # 邻居id列表
        return sorted([agent.unique_id for agent in self.cell.neighborhood.agents])
    def update_payoff(self):
        """
        主要逻辑：
            作为信托者（Investor）的收益：
            遍历对所有邻居的投资 x
            获取从每个邻居收到的返还 y
            计算：(5 - x) + y
            累加所有邻居的收益
            作为受托人（Trustee）的收益：
            遍历对所有邻居的返还 y
            获取邻居对该代理的投资 x
            计算：3x - y
            累加所有邻居的收益
            总收益 = 信托者收益 + 受托者收益
        Returns: self.payoff
        """
        payoff = 0.0
        if self.I_invested_1 and self.I_received_4:
            invest_x_dict = self.I_invested_1[-1]
            receive_y_dict = self.I_received_4[-1]

            for neighbor_id, x in invest_x_dict.items():
                y = receive_y_dict.get(neighbor_id, 0)

                investor_payoff = (5 - x) + y
                payoff += investor_payoff

        if self.T_received_2 and self.T_returned_3:
            invest_x_dict = self.T_received_2[-1]
            receive_y_dict = self.T_returned_3[-1]

            for neighbor_id, x in invest_x_dict.items():
                y = receive_y_dict.get(neighbor_id, 0)

                trustee_payoff = 3 * x - y
                payoff += trustee_payoff

        self.payoff = payoff
        return payoff

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


    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用LLM获取原始响应"""
        pass
