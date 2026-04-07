""""""
import os
import json
import re
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

# 直接复制到代码中使用
system_prompt = """你是中等亲社会倾向的参与者，进行格状网络多邻居多轮对称信任博弈。每轮初始5代币，委托代币×3增值；对每位邻居同时为信任者(0-5整数)、受托人(0至3倍委托值整数)。信任者阶段：受限玩家统一决策，自由玩家独立决策；受托人无约束。依据上轮行为决策，禁止极端自利/利他。输出仅包含邻居ID、阿拉伯数字、英文分号，无多余内容，按邻居顺序输出。"""
output_type = "输出格式：严格按「邻居ID:委托数;邻居ID:委托数;……」的格式输出。"
front = "第{self.step}轮实验; 你的博弈邻居列表：{neighbor_ids};"
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
        # self.neighbor_pairs = self.get_all_neighbor_pairs()
        # 创建LLM模型
        self.llm_model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=self.params.model,
            url=self.params.api_base_url,
            api_key=self.params.api_key,
            model_config_dict={"temperature": self.params.temperature},
        )
        # 创建评价智能体
        self.critic_agent = ChatAgent(
            BaseMessage(
                role_name="critic",
                role_type=RoleType.ASSISTANT,
                meta_dict={},
                content="格式化输出",
            ),
            model=self.llm_model,
            output_language="Chinese",
        )

        # 给agent设置ChatAgent（1、人物 2、根据不同的博弈类型设置初始提示词（不是第一轮））
        self.create_agents()
        self.step:int = 0
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Cooperating_Agents": lambda m: len(
                    [a for a in m.agents if a.strategy == "C"]
                )
            },
            # agent_reporters={
            #     # "Strategy": lambda a: StrategyType.get_strategy_name(a.strategy),
            # }
        )

    def get_all_neighbor_pairs(self) -> list:
        """
        获取网格中所有的邻居对。
        例如 [[0,1,2],[3,4,5],[6,7,8]] 中，0 号节点的邻居对有 [0,1],[0,3],[0,2],[0,6]
        返回格式：[[agent_id_1, agent_id_2], ...]
        """
        neighbor_pairs = []
        width = self.params.width
        height = self.params.height

        # 遍历所有节点
        for x in range(width):
            for y in range(height):
                agent_id = x * height + y

                # 获取该位置的所有邻居（冯·诺依曼邻域：上下左右）
                cell = self.grid[(x, y)]
                neighbors = list(cell.neighborhood.agents)

                # 将当前节点与每个邻居组成对
                for neighbor in neighbors:
                    neighbor_id = neighbor.unique_id
                    # 避免重复，只添加 id 小的在前面的对
                    if agent_id < neighbor_id:
                        neighbor_pairs.append([agent_id, neighbor_id])

        return neighbor_pairs

    def create_agents(self):
        # characters = list(chara_prompt)
        i = 0
        for x in range(self.params.width):
            for y in range(self.params.height):
                agent = BaseAgent(model=self, unique_id=i, cell=self.grid[(x, y)])
                # 设置ChatAgent
                agent.set_llm_agent(
                    ChatAgent(
                        BaseMessage(
                            role_name="player",
                            role_type=RoleType.USER,
                            meta_dict={},
                            content=system_prompt,
                        ),
                        model=self.llm_model,
                        output_language="Chinese",
                    )
                )
                i = i + 1
                # TODO 记录前置固化提示词信息

    def get_agent(self, agent_id):
        """根据ID获取智能体"""
        return self.agents[agent_id]

    def _step(self):
        """模型每一步的执行"""
        self.step += 1
        # print("=" * 20 + f"Step {self.steps}..." + "=" * 20)

        # 1. 两两博弈，决定策略
        is_first_round = True if self.step == 1 else False
        self.play_games(is_first_round)

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

    def play_games(self, is_first_round):

        for i in range(self.params.width * self.params.height):
            focal_agent = self.get_agent(i)
            neighbors = [*list(focal_agent.cell.neighborhood.agents)]
            # 中心玩家作为信托者，发送提示词进行决策
            focal_agent_response = self.investor_call_llm(focal_agent, sorted([neighbor.unique_id for neighbor in neighbors]), is_first_round)
            # 使用轻量级的正则表达式提取，（无需额外API调用）
            investe_amounts = self.extract_decisions_regex(focal_agent_response.content)


            # 中心玩家的邻居作为受托人，依次发送提示词进行决策
            for neighbor in neighbors:
                value = investe_amounts.get(neighbor.unique_id,2)
                feedback_info = f"邻居id_{i}向你(id_{neighbor.unique_id})委托了{value}个代币，根据实验规则，你实际获得{value * 3}个代币。本次仅针对邻居id_{i}做单独返还决策，该决策与其他邻居无任何关联，后续处理其他邻居时将重新发送专属提示词；你决定返还多少代币给你的这位邻居？"
                neighbor_response = self.trustee_call_llm(neighbor, focal_agent, is_first_round,feedback_info)

                # 使用正则表达式提取返还金额
                return_amount = self.extract_decisions_regex(neighbor_response.content)
                # TODO 记录返还金额，用于下一轮给中心玩家的提示词信息
        # TODO 记录本轮博弈信息，提取重要信息用于下一轮博弈

    def investor_call_llm(self, focal_agent, neighbor_ids, is_first_round, feedback_info=""):

        if is_first_round:
            investor_prompt = "投资者阶段-" + front.format(self=self, neighbor_ids=neighbor_ids) + output_type
        else:
            investor_prompt = "投资者阶段-"+ front.format(self=self, neighbor_ids=neighbor_ids) + feedback_info + output_type

        focal_agent_response = focal_agent.llm_agent.step(investor_prompt).msgs[0]
        # TODO 记录向信托者发送的提示词信息
        # TODO 记录信托者的回复信息
        print(f"focal_agent_response: {focal_agent_response.content}")
        return focal_agent_response

    def trustee_call_llm(self, focal_agent, neighbor, is_first_round, feedback_info=""):
        neighbor_ids = [a.unique_id for a in neighbor.cell.agents]  # TODO 返回的邻居id有问题,可以舍弃该信息
        if is_first_round:
            trustee_prompt = "受托人阶段-" + front.format(self=self, neighbor_ids=neighbor_ids) + feedback_info + output_type
        else:
            trustee_prompt = "受托人阶段--" + front.format(self=self, neighbor_ids=neighbor_ids) + feedback_info + output_type


        neighbor_response = neighbor.llm_agent.step(trustee_prompt).msgs[0]
        # TODO 记录向信托者发送的提示词信息
        # TODO 记录信托者的回复信息
        print(f"neighbor_response: {neighbor_response.content}")
        return neighbor_response

    def extract_decisions_regex(self, answer):

        pattern = r"(\d+)\s*:\s*(\d+)"
        matches = re.findall(pattern, answer)

        result = {}
        for neighbor_id, amount in matches:
            result[int(neighbor_id)] = int(amount)

        return result

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
            self._step()

        self.print_final_stats()

    def print_final_stats(self):
        """打印最终统计信息"""
        print(f"\n============================= 运行完成！=================================")
