""""""
import os
from enum import Enum
import numpy as np
from agents import BaseAgent, StrategyType
from params import Params
import typing
import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.examples.advanced.pd_grid.agents import PDAgent
from mesa.experimental.scenarios import Scenario
from camel.models import ModelFactory
from camel.types import ModelPlatformType

class GAME_TYPE(Enum):
    PDG = "pd"
    TRUST = "trust"

class PrisonersDilemmaScenario(Scenario):
    """Scenario for Prisoner's Dilemma model."""
    width: int = 2
    height: int = 2
    payoff: None | dict[tuple[str, str], float] = {
        ("C", "C"): 1,
        ("C", "D"): 0,
        ("D", "C"): 1.6,
        ("D", "D"): 0,
    }
    torus: bool = True

class GameModel(mesa.Model):
    payoff: typing.ClassVar[dict[tuple[str, str], float]] = {
        ("C", "C"): 1,
        ("C", "D"): 0,
        ("D", "C"): 1.6,
        ("D", "D"): 0,
    }
    def __init__(
        self,
        N = 4,    # 节点数量
        game_type=GAME_TYPE.TRUST,
        scenario: PrisonersDilemmaScenario = PrisonersDilemmaScenario,
    ):
        """
        初始化信任博弈模型
        """
        super().__init__(scenario=scenario)

        # 模型参数
        self.params = Params()
        self.N = N
        self.game_type = game_type
        self.grid = OrthogonalMooreGrid(
            (scenario.width, scenario.height), torus=scenario.torus, random=self.random
        )

        if scenario.payoff is not None:
            self.payoff_matrix = scenario.payoff

        # 创建LLM模型
        self.llm_model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=self.params.model,
            url=self.params.api_base_url,
            api_key=self.params.api_key,
            model_config_dict={"temperature": self.params.temperature, "max_tokens": self.params.max_tokens},
        )
        # 创建智能体
        PDAgent.create_agents(
            self, len(self.grid.all_cells.cells), cell=self.grid.all_cells.cells
        )

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Cooperating_Agents": lambda m: len(
                    [a for a in m.agents if a.move == "C"]
                )
            },
            # agent_reporters={
            #     # "Strategy": lambda a: StrategyType.get_strategy_name(a.strategy),
            # }
        )

        self.running = True
        self.datacollector.collect(self)


    def get_agent(self, agent_id):
        """根据ID获取智能体"""
        return self.agents[agent_id]

    def step(self):
        """模型每一步的执行"""
        # print("=" * 20 + f"Step {self.steps}..." + "=" * 20)

        # 1. 更新策略
        self.agents.do("decide")

        # 5. 计算收益
        self.agents.do("update_payoff")

        # 6. 收集数据
        self.datacollector.collect(self)

        # 7. 打印step进度信息
        # if self.steps % 50 == 0 or self.steps <= 10:
        #     print(
        #         f"Step {self.steps}: "
        #         f"Average Payoff = {self.get_avg_payoff()}, "
        #         f"Global Payoff = {self.get_global_payoff()}, "
        #         f"Strategy Proportion = {self.get_strategy_proportion('IT')}, "
        #         f"{self.get_strategy_proportion('IU')}, "
        #         f"{self.get_strategy_proportion('NT')}, "
        #         f"{self.get_strategy_proportion('NU')}"
        #     )

    # 数据收集器所需的模型报告函数
    def get_avg_payoff(self):
        """计算平均累计收益"""
        payoffs = [agent.payoff for agent in self.agents]
        return round(np.mean(payoffs), 4) if payoffs else 0.0

    def get_strategy_proportion(self, strategy_type):
        """获取策略比例"""
        count = sum(1 for agent in self.agents if agent.strategy == strategy_type)
        return round(count / self.N, 4) if self.N > 0 else 0.0


    def run_model(self, max_steps=100):
        """运行模型指定步数"""
        self.max_steps = max_steps
        for i in range(max_steps):
            self.step()

        self.print_final_stats()

    def print_final_stats(self):
        """打印最终统计信息"""
        print(f"\n最终统计:")
        # print(f"  平均累计收益: {self.get_avg_payoff():.4f}")
        print(f"  全局收益总和: {self.get_global_payoff():.4f}")
        # print(f"  总成对博弈次数: {sum(len(a.pairwise_games_payoffs) for a in self.agents)}")
        # print(f"  总群组博弈次数: {sum(len(a.group_games_payoffs) for a in self.agents)}")
        # print(f"  总策略更新次数: {sum(len(a.strategy_history) for a in self.agents)}")

        print(f"\n最终策略分布:")
        print(f"  IT: {self.get_strategy_proportion(StrategyType.IT):.3f} "
              f"({sum(1 for a in self.agents if a.strategy == StrategyType.IT)}个)")
        print(f"  IU: {self.get_strategy_proportion(StrategyType.IU):.3f} "
              f"({sum(1 for a in self.agents if a.strategy == StrategyType.IU)}个)")
        print(f"  NT: {self.get_strategy_proportion(StrategyType.NT):.3f} "
              f"({sum(1 for a in self.agents if a.strategy == StrategyType.NT)}个)")
        print(f"  NU: {self.get_strategy_proportion(StrategyType.NU):.3f} "
              f"({sum(1 for a in self.agents if a.strategy == StrategyType.NU)}个)")

