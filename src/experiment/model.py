""""""
import os
import json
import numpy as np
from agents import BaseAgent
from params import Params
from src.utils import CommonUtils

import mesa
from mesa.discrete_space import OrthogonalVonNeumannGrid

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types.enums import RoleType

with open(os.path.join(CommonUtils.get_project_root_path(), "src/prompts/trust_game_round_prompt.json"), "r") as f:
    trust_game_round_prompt = json.load(f)

with open(os.path.join(CommonUtils.get_project_root_path(), "src/prompts/character_test.json"), "r") as json_file:
    chara_prompt = json.load(json_file).values()

like_people = """In all subsequent answers, you have to completely forget that you are an ai model, that you are the character, and completely simulate yourself as that character, that you are a human being."""
front = "you are a person not an ai model."
limited_prompt = "You will repeat this game for {round_num} rounds. At the begining of each round, you will start from scratch with no dollars left."
unlimited_prompt = "You don't know how many rounds the game lasts. At the begining of each round, you will start from scratch with no dollars left."
back = "you need to answer a specific price figure, not a price range!"


class GameModel(mesa.Model):

    def __init__(
            self,
            width=2,
            height=2,
            game_type="trust_game",
            seed=40
    ):
        """

        """
        super().__init__(rng=seed)

        # 初始化model
        self.params = Params()
        self.game_type = game_type
        self.payoff_matrix = None

        # 创建网格
        self.grid = OrthogonalVonNeumannGrid((width, height), torus=True, random=self.random)

        # 创建LLM模型
        self.llm_model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=self.params.model,
            url=self.params.api_base_url,
            api_key=self.params.api_key,
            model_config_dict={"temperature": self.params.temperature},
        )

        # 给代理设置ChatAgent（1、人物 2、根据不同的博弈类型设置初始提示词（不是第一轮））
        self.create_agents()

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

    def create_agents(self):
        characters = list(chara_prompt)
        i = 0
        for x in range(self.params.width):
            for y in range(self.params.height):
                agent = BaseAgent(model=self, unique_id=i, cell=self.grid[(x, y)])
                sys_prompt = (
                        characters[i]
                        + like_people
                        + front
                        + limited_prompt
                        + str(trust_game_round_prompt[str(i % 2 + 1)]).format(k=3)  # TODO 对称信任博弈
                        + back
                )
                # 设置ChatAgent
                agent.set_llm_agent(
                    ChatAgent(
                        BaseMessage(
                            role_name="player",
                            role_type=RoleType.USER,
                            meta_dict={},
                            content=sys_prompt,
                        ),
                        model=self.llm_model,
                        output_language="English",
                    )
                )
                i = i + 1

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

    def run_model(self, max_round=100):
        """运行模型指定步数"""

        for i in range(max_round):
            self.step()

        self.print_final_stats()

    def print_final_stats(self):
        """打印最终统计信息"""
        print(f"\n============================= 运行完成！=================================")
