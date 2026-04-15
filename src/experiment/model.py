""""""
import os
import json
import re
import numpy as np
import logging
from numpy.random import random_integers

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
    all_prompts = json.load(f)

p_characters = all_prompts.get("Character", [])
p_like_people = all_prompts.get("Like-people", "")
p_experiment_info = all_prompts.get("Experiment info", "")
p_game_rules = all_prompts.get("Game rule", "")
p_position_template = all_prompts.get("Position", "")
p_player_restrictions = all_prompts.get("Player restrictions", "")
p_ensure = all_prompts.get("Ensure", "")
p_decision = all_prompts.get("Decision", "")
p_memory = all_prompts.get("Memory", "")
p_decision_stages = all_prompts.get("Decision stages", "")
p_output_requirements = all_prompts.get("Output requirements", "")
p_system_prompt = p_like_people + p_experiment_info + p_game_rules.get("trust game")


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
        self.step: int = 0

        self.agents_invested_amounts = {}  # 记录所有代理的投资金额
        self.agents_returned_amounts = {}  # 记录所有代理的返还金额

        # 初始化记录数据结构
        self.round_input_record = {}
        self.round_output_record = {}

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
        initial_sys_prompt = {}
        for i in range(self.num_agents):
            agent = self.get_agent(i)
            agent.unique_id = agent.unique_id - 1

            character = p_characters[i]
            position = p_position_template.format(id=agent.unique_id, coordinate=agent.cell.coordinate, neighbors=agent.neighbor_ids)
            content: str = character + p_system_prompt + position + p_player_restrictions.get("Free player") + p_ensure
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
            initial_sys_prompt[f"agent_{agent.unique_id}_position_{agent.cell.coordinate}"] = content

        self.save_records(initial_sys_prompt)

    def _step(self):
        """模型每一步的执行"""
        self.step += 1

        # 1. 依次扮演信托者
        self.role_stage(player_type="investor")

        # 2. 整理所有信托者的回复
        self.record_agent_investment()

        # 3. 依次扮演受托人
        self.role_stage(player_type="trustee")

        # 4. 整理所有受托人的回复
        self.record_agent_return()

        # 5. 计算收益
        self.agents.do("update_payoff")

        self.save_records()

        # 6. 重置记录
        self.round_input_record.clear()
        self.round_output_record.clear()

        # 7. 收集数据
        # self.datacollector.collect(self)

    def role_stage(self, player_type="investor"):

        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 第 {self.step} 轮博弈 - {player_type} 阶段开始")
        logger.info(f"{'=' * 60}")

        agent_input = {}
        agent_output = {}
        for i in range(self.num_agents):
            focal_agent = self.get_agent(i)

            add_info = ''
            prompt = self.prepare_prompt(agent=focal_agent, player_type=player_type, additional_info=add_info)
            # focal_agent_response = self._call_llm(focal_agent=focal_agent, prompt=investor_prompt, player_type="Investor")
            # 使用轻量级的正则表达式提取，（无需额外API调用）
            invest_value = {}
            for id in focal_agent.neighbor_ids:
                invest_value[id] = random_integers(0, 5)
            focal_agent_response = f"作为 Emily，一名习惯用逻辑和算法思考的软件工程师，我开始分析这一轮的游戏策略。这是一个对称的信任博弈，处于多轮实验的初期阶段。虽然第一轮没有历史数据可以参考，但在重复博弈的背景下，建立互惠的合作规范对于最大化长期总收益至关重要。\n从数学期望来看，如果我不发送任何代币（x=0），虽然能确保不亏损，但也无法获得信托投资带来的潜在增值回报，这会破坏合作的可能性。反之，如果发送全部 5 个代币，风险过高。考虑到我是一个追求卓越的分析师，我会选择一个既能有效传递信任信号，又能保留一定安全边际的数量。发送 4 个代币意味着我将大部分资源投入到合作伙伴手中，这向邻居展示了强烈的合作意愿，同时也留下了一个代币作为风险缓冲。鉴于目前所有邻居的情况相同，且我是自由玩家，可以对每个邻居独立决策，但为了确立一致的社交规范，我对这两个邻居将采取相同的策略。\nFinally, I decide to send {invest_value} to each neighbor."
            send_amounts = self.extract_decisions_regex(focal_agent_response)

            if player_type == "investor":
                focal_agent.I_invested_1.append(send_amounts)
                self.agents_invested_amounts[focal_agent.unique_id] = invest_value
            elif player_type == "trustee":
                focal_agent.T_returned_3.append(send_amounts)
                self.agents_returned_amounts[focal_agent.unique_id] = invest_value
            # 记录投资者输入提示词和输出回复
            agent_input[f"agent_{focal_agent.unique_id}"] = prompt
            agent_output[f"agent_{focal_agent.unique_id}"] = focal_agent_response

        if player_type == "investor":
            self.round_input_record["as_investor"] = agent_input
            self.round_output_record["as_investor"] = agent_output
        elif player_type == "trustee":
            self.round_input_record["as_trustee"] = agent_input
            self.round_output_record["as_trustee"] = agent_output

        logger.info(f"✅ 第 {self.step} 轮投资者阶段完成\n")

    def play_trustee(self, is_first_round, round_key):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 第 {self.step} 轮博弈 - 受托人阶段开始")
        logger.info(f"{'=' * 60}")

        agent_input = {}
        agent_output = {}
        for i in range(self.num_agents):
            focal_agent = self.get_agent(i)

            add_info = ''
            trustee_prompt = self.prepare_prompt(is_first_round=is_first_round, additional_info=add_info)
            focal_agent_response = self._call_llm(focal_agent=focal_agent, prompt=trustee_prompt, player_type="Trustee")
            # 使用轻量级的正则表达式提取，（无需额外API调用）
            returned_amounts = self.extract_decisions_regex(focal_agent_response.content)

            # focal_agent.T_returned_amounts.append(returned_amounts)
            # self.agents_returned_amounts.append(returned_amounts)

            # 记录投资者输入提示词和输出回复
            agent_input[f"agent_{focal_agent.unique_id}"] = trustee_prompt
            agent_output[f"agent_{focal_agent.unique_id}"] = focal_agent_response
        self.round_input_record["as_trustee"] = agent_input
        self.round_output_record["as_trustee"] = agent_output

        logger.info(f"✅ 第 {self.step} 轮受托人阶段完成\n")

    def prepare_prompt(self, agent, player_type, additional_info="") -> str:

        prompt = additional_info
        output_r = p_output_requirements.get("Format")
        if self.step == 1:
            if player_type == "investor":
                decision_stages = p_decision_stages.get("investor")
                prompt = decision_stages + output_r
            elif player_type == "trustee":
                received_amounts = agent.T_received_2[-1]
                investor_decision = p_decision.format(step=self.step, received_amounts=received_amounts)
                decision_stages = p_decision_stages.get("trustee")
                prompt = investor_decision + decision_stages + output_r

        else:
            if player_type == "investor":
                memory = p_memory.format(I_invested_1=agent.I_invested_1, I_received_4=agent.I_received_4, T_received_2=agent.T_received_2, T_returned_3=agent.T_returned_3, payoff=agent.payoff)
                decision_stages = p_decision_stages.get("investor")
                prompt = memory + decision_stages + output_r
            elif player_type == "trustee":
                received_amounts = agent.T_received_2[-1]
                investor_decision = p_decision.format(step=self.step, received_amounts=received_amounts)
                decision_stages = p_decision_stages.get("trustee")
                prompt = investor_decision + decision_stages + output_r

        return prompt

    def _call_llm(self, focal_agent, prompt, player_type="Investor"):
        import time
        start_time = time.time()
        focal_agent_response = focal_agent.llm_agent.step(prompt).msgs[0]
        elapsed_time = time.time() - start_time

        logger.info(f"✅ [{player_type}] Agent {focal_agent.unique_id} 投资决策完成 - API耗时: {elapsed_time:.2f}秒")
        logger.info(f"  响应: {focal_agent_response.content[:100]}...")

        return focal_agent_response

    def record_agent_investment(self):
        for i in range(self.num_agents):
            focal_agent = self.get_agent(i)

            received_from_neighbors = {}
            for neighbor_id in focal_agent.neighbor_ids:
                neighbor_agent = self.get_agent(neighbor_id)

                if neighbor_agent.I_invested_1:
                    last_investment = neighbor_agent.I_invested_1[-1]

                    amount_to_focal = last_investment.get(focal_agent.unique_id, 0)

                    received_from_neighbors[neighbor_id] = amount_to_focal

            focal_agent.T_received_2.append(received_from_neighbors)

            logger.debug(f"Agent {i} 收到邻居投资: {received_from_neighbors}")

        # self.agents_invested_amounts = []

    def record_agent_return(self):
        for i in range(self.num_agents):
            focal_agent = self.get_agent(i)

            returned_from_neighbors = {}
            for neighbor_id in focal_agent.neighbor_ids:
                neighbor_agent = self.get_agent(neighbor_id)

                if neighbor_agent.T_returned_3:
                    last_return = neighbor_agent.T_returned_3[-1]

                    amount_to_focal = last_return.get(focal_agent.unique_id, 0)

                    returned_from_neighbors[neighbor_id] = amount_to_focal

            focal_agent.I_received_4.append(returned_from_neighbors)

            logger.info(f"Agent {i} 收到邻居返还: {returned_from_neighbors}")

    def extract_decisions_regex(self, answer):
        pattern = r"Finally,\s*I\s+decide\s+to\s+send\s*\[([^\]]+)\]"
        match = re.search(pattern, answer, re.IGNORECASE)

        if not match:
            logger.warning(f"未找到标准格式，尝试备用提取方式。原始响应: {answer[:100]}")
            return self._extract_fallback(answer)

        content = match.group(1)
        pairs = re.findall(r"(\d+)\s*:\s*(\d+)", content)

        if not pairs:
            logger.warning(f"找到格式但未解析到键值对: {content}")
            return {}

        result = {}
        for neighbor_id, amount in pairs:
            result[int(neighbor_id)] = min(int(amount), 5)

        logger.debug(f"提取投资决策: {result}")
        return result

    def _extract_fallback(self, answer):
        pairs = re.findall(r"(\d+)\s*:\s*(\d+)", answer)
        if not pairs:
            logger.error(f"无法从响应中提取任何决策: {answer[:200]}")
            return {}

        result = {}
        for neighbor_id, amount in pairs:
            nid, amt = int(neighbor_id), int(amount)
            if 0 <= amt <= 5:
                result[nid] = amt

        return result

    def save_records(self, data=None):
        """保存记录到JSON文件"""
        results_dir = os.path.join(CommonUtils.get_project_root_path(), "results")
        os.makedirs(results_dir, exist_ok=True)
        record_file = os.path.join(results_dir, "record_model_test.json")

        if self.step == 0:
            record_data = {
                "initial_sys_prompt": data,
                "input_record": {},
                "output_record": {}
            }
            with open(record_file, "w", encoding="utf-8") as f:
                json.dump(record_data, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 初始化记录文件: {record_file}")
        else:
            with open(record_file, "r", encoding="utf-8") as f:
                record_data = json.load(f)

            record_data["input_record"][f"round_{self.step}"] = self.round_input_record
            record_data["output_record"][f"round_{self.step}"] = self.round_output_record

            with open(record_file, "w", encoding="utf-8") as f:
                json.dump(record_data, f, ensure_ascii=False, indent=2)

            logger.debug(f"✅ 第 {self.step} 轮记录已保存到文件")

    def run_model(self, max_round=100):
        """运行模型指定步数"""
        import tqdm
        for i in tqdm.trange(max_round):
            self._step()

        self.print_final_stats()

    def get_agent(self, agent_id):
        """根据ID获取智能体"""
        return self.agents[agent_id]

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
