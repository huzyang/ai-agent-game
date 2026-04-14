""""""
import os
import json
import re
import numpy as np
import logging
from agents import BaseAgent
from params import Params, ModelType, GameType
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

with open(os.path.join(CommonUtils.get_project_root_path(), "src/prompts/all_prompts.json"), "r", encoding="utf-8") as f:
    all_prompt_list = json.load(f)

all_prompts = {}
for item in all_prompt_list:
    for key, value in item.items():
        all_prompts[key] = value

p_characters = all_prompts.get("Character", [])
p_like_people = all_prompts.get("Like-people", "")
p_experiment_info = all_prompts.get("Experiment info", "")
p_game_rules = all_prompts.get("Game rule", [])
p_position_template = all_prompts.get("Position", "")
p_player_restrictions = all_prompts.get("Player restrictions", [])
p_ensure = all_prompts.get("Ensure", "")
p_decision_memory = all_prompts.get("Decision and memory", [])
p_decision_stages = all_prompts.get("Decision stages", [])
p_output_requirements = all_prompts.get("Output requirements", [])
p_system_prompt = p_like_people + p_experiment_info + p_game_rules[0].get("trust game")

logger = logging.getLogger(__name__)
class GameScenario(Scenario):
    """Scenario for model."""
    num_agents: int = 4  # 节点数
    width: int = int(num_agents ** 0.5)  # 根号 N
    height: int = width
    model_type: str = ModelType.QWEN3_5_FLASH.value
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
        self.num_agents = scenario.num_agents
        self.model_type = scenario.model_type
        self.game_type = scenario.game_type

        # 初始化记录数据结构
        self.agents_invested_amounts = []
        self.agents_returned_amounts = []

        self.initial_sys_prompt = []
        self.input_record = {}
        self.output_record = {}

        # 创建网格
        self.grid = OrthogonalVonNeumannGrid((scenario.width, scenario.height), torus=True, random=self.random)
        BaseAgent.create_agents(self, n=self.num_agents, cell=self.grid.all_cells.cells)

        # 创建LLM模型
        self.llm_model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=self.model_type,
            url=self.params.api_base_url,
            api_key=self.params.api_key,
            model_config_dict={"temperature": self.params.temperature},
        )

        # 创建智能体、给智能体设置ChatAgent（1、人物 2、根据不同的博弈类型设置初始提示词（不是第一轮））
        self.init_chat_agent()
        self.step: int = 0
        # self.datacollector = mesa.DataCollector(
        #     agent_reporters={
        #         "Invested amount": lambda a: a.invested_amounts,
        #         "Received amount": lambda a: a.received_amounts,
        #     }
        # )

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

    def init_chat_agent(self):
        for i in range(self.num_agents):
            agent = self.get_agent(i)

            character = p_characters[i]
            position = f"Your ID: {agent.unique_id}. Your position in the lattice: {agent.cell.coordinate}. Your neighbors in the lattice network: {agent.neighbor_ids}."
            content:str = character+p_system_prompt+position+p_player_restrictions[1].get("Free payer")+p_ensure
            # 设置ChatAgent
            agent.set_llm_agent(
                ChatAgent(
                    BaseMessage(
                        role_name="player",
                        role_type=RoleType.USER,
                        meta_dict={},
                        content=content
                    ),
                    model=self.llm_model,
                    output_language="Chinese",
                )
            )

            # 记录人物特征与初始系统提示词 initial_sys_prompt
            self.initial_sys_prompt.append({
                f"agent_{agent.unique_id}_position_{agent.cell.coordinate}": content
            })

    def get_agent(self, agent_id):
        """根据ID获取智能体"""
        return self.agents[agent_id]

    def _step(self):
        """模型每一步的执行"""
        self.step += 1
        is_first_round = True if self.step == 1 else False
        round_key = f"round_{self.step}"
        if round_key not in self.input_record:
            self.input_record[round_key] = []
        if round_key not in self.output_record:
            self.output_record[round_key] = []

        # 1. 依次扮演信托者
        self.play_investor(is_first_round,round_key)

        # 2. 整理所有信托者的回复
        self.record_agent_investment()
        # 3. 依次扮演受托人
        self.play_trustee(is_first_round,round_key)

        # 4. 整理所有受托人的回复
        self.record_agent_return()

        # 5. 计算收益
        self.agents.do("update_payoff")

        # 6. 每轮结束后保存记录
        # self.save_records()

        # 7. 收集数据
        # self.datacollector.collect(self)

    def play_investor(self, is_first_round, round_key):

        round_input = {"agent_" + str(i + 1): [] for i in range(self.params.width * self.params.height)}
        round_output = {"agent_" + str(i + 1): [] for i in range(self.params.width * self.params.height)}

        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 第 {self.step} 轮博弈 - 投资者阶段开始")
        logger.info(f"{'=' * 60}")

        for i in range(self.num_agents):
            focal_agent = self.get_agent(i)

            add_info = ''
            investor_prompt = self.prepare_prompt(is_first_round=is_first_round, additional_info=add_info)
            focal_agent_response = self._call_llm(focal_agent=focal_agent, prompt=investor_prompt, player_type="Investor")
            # 使用轻量级的正则表达式提取，（无需额外API调用）
            invested_amounts = self.extract_decisions_regex(focal_agent_response.content)

            focal_agent.I_invested_amounts.append(invested_amounts)
            # 记录投资者输入提示词和输出回复
            round_input[f"agent_{i}"].append({f"as_investor": investor_prompt})
            round_output[f"agent_{i}"].append({f"as_investor": focal_agent_response.content})

        self.input_record[round_key].append({"investor_input": round_input})
        self.output_record[round_key].append({"investor_output": round_output})

        logger.info(f"✅ 第 {self.step} 轮投资者阶段完成\n")

    def play_trustee(self, is_first_round, round_key):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 第 {self.step} 轮博弈 - 受托人阶段开始")
        logger.info(f"{'=' * 60}")


    def prepare_prompt(self, is_first_round, additional_info="") -> str:
        game_stage = p_decision_stages[0].get("As investor")
        output_r = p_output_requirements[1].get("Format")
        if is_first_round:
            investor_prompt = game_stage + output_r
        else:
            investor_prompt = additional_info + game_stage + output_r
        return investor_prompt

    def _call_llm(self, focal_agent, prompt, player_type="Investor"):
        import time
        start_time = time.time()
        focal_agent_response = focal_agent.llm_agent.step(prompt).msgs[0]
        elapsed_time = time.time() - start_time

        logger.info(f"✅ [{player_type}] Agent {focal_agent.unique_id} 投资决策完成 - API耗时: {elapsed_time:.2f}秒")
        logger.info(f"  响应: {focal_agent_response.content[:100]}...")
        return focal_agent_response

    def record_agent_investment(self):
        # for i in range(self.params.width * self.params.height):
        #     focal_agent = self.get_agent(i)
        #     focal_agent.I_received_amounts.append(focal_agent.I_invested_amounts[-1])
        #     focal_agent.I_invested_amounts = []
        pass
    def record_agent_return(self):
        # for i in range(self.params.width * self.params.height):
        #     focal_agent = self.get_agent(i)
        pass

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
        import tqdm
        for i in tqdm.trange(max_round):
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
