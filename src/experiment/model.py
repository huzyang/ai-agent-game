""""""
import os
import json
import re
import numpy as np

from agents import BaseAgent
from params import Params,ModelType,GameType
from src.utils import CommonUtils
import multiprocessing as mp

import mesa
from mesa.discrete_space import OrthogonalVonNeumannGrid
from mesa.experimental.scenarios import Scenario

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types.enums import RoleType, logger

with open(os.path.join(CommonUtils.get_project_root_path(), "src/prompts/all_prompts.json"), "r") as f:
    all_prompt_list = json.load(f)

all_prompts = {}
for item in all_prompt_list:
    for key, value in item.items():
        all_prompts[key] = value


character_prompts = all_prompts.get("Character", [])
like_people_prompt = all_prompts.get("Like-people", "")
experiment_info = all_prompts.get("Experiment info", "")
game_rules = all_prompts.get("Game rule", [])
position_template = all_prompts.get("Position", "")
player_restrictions = all_prompts.get("Player restrictions", [])
decision_memory = all_prompts.get("Decision and memory", [])
decision_stages = all_prompts.get("Decision stages", [])
output_requirements = all_prompts.get("Output requirements", [])


class GameScenario(Scenario):
    """Scenario for model."""
    num_agents: int = 4  # 节点数
    width: int = int(num_agents ** 0.5)  # 根号 N
    height: int = width
    model_type: str = ModelType.QWEN3_5_FLASH.value,
    game_type: str = GameType.TRUST.value


class GameModel(mesa.Model):

    def __init__(
            self,
            scenario: GameScenario = GameScenario,
    ):
        """

        """
        super().__init__(scenario=scenario)

        # 初始化model
        self.params = Params()
        self.model_type = scenario.model_type
        self.game_type = scenario.game_type

        # 初始化记录数据结构
        self.initial_sys_prompt = []
        self.input_record = {}
        self.output_record = {}

        # 创建网格
        self.grid = OrthogonalVonNeumannGrid(
            (scenario.width, scenario.height), torus=True, random=self.random
        )

        # 创建LLM模型
        self.llm_model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=self.model_type,
            url=self.params.api_base_url,
            api_key=self.params.api_key,
            model_config_dict={"temperature": self.params.temperature},
        )

        # 创建智能体、给智能体设置ChatAgent（1、人物 2、根据不同的博弈类型设置初始提示词（不是第一轮））
        self.create_agents()
        self.step: int = 0
        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Invested amount": lambda a: a.invested_amounts,
                "Received amount": lambda a: a.received_amounts,
            }
        )

        # 创建评价智能体
        # self.critic_agent = ChatAgent(
        #     BaseMessage(
        #         role_name="critic",
        #         role_type=RoleType.ASSISTANT,
        #         meta_dict={},
        #         content="格式化输出",
        #     ),
        #     model=self.llm_model,
        #     output_language="Chinese",
        # )

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

                character_info = character_prompts[i]
                system_prompt = f"{character_info}\n\n{like_people_prompt}\n\n{experiment_info}\n\n{game_rules[0]}"

                # 记录人物特征与初始系统提示词 initial_sys_prompt
                # self.initial_sys_prompt.append({
                #     f"agent_{i}_cell({x},{y})": system_prompt
                # })
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

    def get_agent(self, agent_id):
        """根据ID获取智能体"""
        return self.agents[agent_id]

    def _step(self):
        """模型每一步的执行"""
        self.step += 1
        is_first_round = True if self.step == 1 else False

        # 1. 依次扮演信托者
        self.play_investor(is_first_round)

        # 2. 依次扮演受托人
        self.play_trustee(is_first_round)

        # 3. 计算收益
        self.agents.do("update_payoff")

        # 4. 收集数据
        # self.datacollector.collect(self)
    def play_investor(self, is_first_round):
        # 1. 准备提示词
        investor_prompt = self.prepair_prompt(is_first_round)

        # 2.调用大模型，处理回复并记录
        pass

    def play_trustee(self, is_first_round):
        # 1. 准备提示词
        # 2.使用多线程调用大模型，处理回复并记录
        n_processes = min(mp.cpu_count() / 2, self.params.width * self.params.height)
        logger.info(f'使用 {n_processes} 个进程并行运行模拟...')
        pass

    def prepair_prompt(self,is_first_round,feedback_info="") -> str:
        if is_first_round:
            investor_prompt = ""
        else:
            investor_prompt = ""
        return investor_prompt

    def play_games(self, is_first_round):
        round_key = f"round_{self.step}"

        if round_key not in self.input_record:
            self.input_record[round_key] = []
        if round_key not in self.output_record:
            self.output_record[round_key] = []

        round_input = {"agent_" + str(i): [] for i in range(self.params.width * self.params.height)}
        round_output = {"agent_" + str(i): [] for i in range(self.params.width * self.params.height)}
        # 中心玩家作为投资者，发送提示词进行决策
        print(f"\n{'=' * 60}")
        print(f"📊 第 {self.step} 轮博弈开始")
        print(f"{'=' * 60}")

        for i in range(self.params.width * self.params.height):

            focal_agent = self.get_agent(i)
            # 中心玩家作为信托者，发送提示词进行决策
            investor_input_prompt, focal_agent_response = self.investor_call_llm(focal_agent, focal_agent.neighbor_ids, is_first_round)
            # 使用轻量级的正则表达式提取，（无需额外API调用）
            investe_amounts = self.extract_decisions_regex(focal_agent_response.content)
            focal_agent.invested_amounts.append(investe_amounts)

            # 记录投资者输入提示词
            # investor_input_prompt = "投资者阶段-" + front.format(self=self, neighbor_ids=sorted_neighbor_ids) + output_type
            round_input[f"agent_{i}"].append({f"investor_agent_{i}": investor_input_prompt})
            # 记录投资者输出回复
            round_output[f"agent_{i}"].append({f"investor_agent_{i}": focal_agent_response.content})

            # 中心玩家的邻居作为受托人，依次发送提示词进行决策
            for neighbor_id in focal_agent.neighbor_ids:
                trustee_agent = self.get_agent(neighbor_id)
                value = investe_amounts.get(neighbor_id, 2)
                feedback_info = f"邻居(id_{i})向你(id_{neighbor_id})委托了{value}个代币，根据实验规则，你实际获得{value * 3}个代币。本次仅针对该邻居(id_{i})做单独返还决策，该决策与其他邻居无任何关联，后续处理其他邻居时将重新发送专属提示词；你决定返还多少代币给你的这位邻居？"

                trustee_input_prompt, trustee_agent_response = self.trustee_call_llm(focal_agent.unique_id, trustee_agent, is_first_round, feedback_info)
                # 使用正则表达式提取返还金额
                return_amount = self.extract_decisions_regex(trustee_agent_response.content)
                # 中心玩家记录返还信息
                focal_agent.received_amounts.append(return_amount)

                # 记录受托人输入
                round_input[f"agent_{neighbor_id}"].append({f"trustee_agent_{i}": trustee_input_prompt})
                # 记录受托人输出
                round_output[f"agent_{neighbor_id}"].append({f"trustee_agent_{i}": trustee_agent_response.content})

        self.input_record[round_key].append(round_input)
        self.output_record[round_key].append(round_output)

        print(f"{'=' * 60}")
        print(f"✅ 第 {self.step} 轮博弈完成")
        print(f"{'=' * 60}\n")

        # 每轮结束后保存记录
        self.save_records()

    def _call_llm(self, focal_agent, investor_prompt):
        import time
        start_time = time.time()
        focal_agent_response = focal_agent.llm_agent.step(investor_prompt).msgs[0]
        elapsed_time = time.time() - start_time

        print(f"\n  🔹 [投资者] Agent {focal_agent.unique_id}")
        print(f"  ├─ 提示词: {investor_prompt}")
        print(f"  ├─ 响应: {focal_agent_response.content}")
        print(f"  └─ API耗时: {elapsed_time:.2f}秒")

        return investor_prompt, focal_agent_response

    def extract_decisions_regex(self, answer):

        pattern = r"(\d+)\s*:\s*(\d+)"
        matches = re.findall(pattern, answer)

        result = {}
        for neighbor_id, amount in matches:
            result[int(neighbor_id)] = int(amount)

        return result

    def save_records(self):
        """保存记录到JSON文件"""
        results_dir = os.path.join(CommonUtils.get_project_root_path(), "results")
        os.makedirs(results_dir, exist_ok=True)

        record_data = [
            {
                "initial_sys_prompt": self.initial_sys_prompt
            },
            {
                "input_record": self.input_record
            },
            {
                "output_record": self.output_record
            }
        ]

        record_file = os.path.join(results_dir, "record_model_test.json")
        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(record_data, f, ensure_ascii=False, indent=2)

    def run_model(self, max_round=100):
        """运行模型指定步数"""

        for i in range(max_round):
            self._step()

        self.print_final_stats()

    def print_final_stats(self):
        """打印最终统计信息"""
        """打印最终统计信息"""
        print(f"\n{'=' * 70}")
        print(f"🎉 模型运行完成！")
        print(f"{'=' * 70}")
        print(f"  • 总轮数: {self.step}")
        print(f"  • 网格大小: {self.params.width} x {self.params.height}")
        print(f"  • 智能体总数: {len(self.agents)}")
        print(f"  • 记录文件: results/record_model_test.json")
        print(f"{'=' * 70}\n")
