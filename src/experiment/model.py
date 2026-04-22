""""""
import os
import json
import re
import sys
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types.enums import RoleType

import mesa
from mesa.discrete_space import OrthogonalVonNeumannGrid
from agents import BaseAgent
from params import Params, GameScenario
from src.utils import CommonUtils

logger = logging.getLogger('experiment')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setStream(sys.stdout)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

with open(os.path.join(CommonUtils.get_project_root_path(), "src/prompts/exp_prompts.json"), "r", encoding="utf-8") as f:
    exp_prompts = json.load(f)

p_characters = exp_prompts.get("Character", [])
random.shuffle(p_characters)
p_characters_student = exp_prompts.get("Character_student", [])
p_like_people = exp_prompts.get("Like-people", "")
p_experiment_info = exp_prompts.get("Experiment_context", "")  # format id, position, neighbors
p_game_rules = exp_prompts.get("Game_rule", "")
p_behavioral_objective = exp_prompts.get("Behavioral_objective", "")
p_output_requirements = exp_prompts.get("Output requirements", "")  # neighbors
p_consistency = exp_prompts.get("Consistency", "")

p_decision_stages = exp_prompts.get("Decision stages", "")  # step, type
p_end = exp_prompts.get("End", "")
p_decision = exp_prompts.get("Decision", "")  # step, received_amounts
p_memory = exp_prompts.get("Memory", "")  # I_invested_1,I_received_4,T_received_2,T_returned_3,payoff


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
        self.height = scenario.height
        self.width = scenario.width
        self.model_type = scenario.model_type
        self.game_type = scenario.game_type
        self.proportion = scenario.proportion
        self.step: int = 0
        self.run_id = scenario.run_id
        self.iteration = scenario.iteration

        self.agents_invested_amounts = {}  # 记录所有代理的投资金额
        self.agents_returned_amounts = {}  # 记录所有代理的返还金额

        # 初始化记录数据结构
        self.initial_sys_prompt = {}
        self.dialogs = {}  # 存储每轮次对话数据
        self.all_data = []  # 存储所有轮次的数据行

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
        width = self.width
        height = self.height

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
        # 计算自由玩家数量
        num_free_players = int(self.num_agents * self.proportion)

        # 创建限制类型列表：前num_free_players个为自由玩家，其余为受限玩家
        restriction_set = []
        for i in range(self.num_agents):
            if i < num_free_players:
                restriction_set.append("free")
            else:
                restriction_set.append("constrained")

        # 随机打乱顺序以随机分配玩家类型
        random.shuffle(restriction_set)

        for i in range(self.num_agents):
            agent = self.get_agent(i)
            agent.unique_id = agent.unique_id - 1

            # 系统提示词组成： p_characters[i] + p_like_people + p_experiment_info + p_game_rules.get("trust_game") + p_behavioral_objective + p_output_requirements + p_consistency
            character = p_characters[i]
            # character = p_characters_student[0]  # 测试使用
            position = p_experiment_info.format(id=agent.unique_id, position=agent.cell.coordinate, neighbors=agent.neighbor_ids)
            output_requirements = p_output_requirements.get("General") + p_output_requirements.get("Simple")
            content: str = character + p_like_people + position + p_game_rules.get("trust_game") + p_behavioral_objective + output_requirements + p_consistency

            agent.type_restriction = restriction_set[i]
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
                    output_language="English",
                )
            )

            # 记录人物特征与初始系统提示词 initial_sys_prompt
            self.initial_sys_prompt[f"agent_{agent.unique_id}_{agent.type_restriction}_position-{agent.cell.coordinate}"] = content

    def _step(self):
        """模型每一步的执行"""
        self.step += 1
        # 初始化当前轮次的记录字典
        round_key = f"round_{self.step}"
        self.dialogs[round_key] = {}

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

        # 6：收集当前轮次数据
        self._collect_data()

        # 7. 收集数据
        # self.agents.do("reset_record")

    def role_stage(self, player_type="investor"):

        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 实验共{self.params.rounds}轮，第 {self.step} 轮博弈 - {player_type} 阶段开始")
        logger.info(f"{'=' * 60}")

        round_key = f"round_{self.step}"
        if round_key not in self.dialogs:
            self.dialogs[round_key] = {}

        # 准备所有任务的参数
        tasks = []
        for i in range(self.num_agents):
            focal_agent = self.get_agent(i)
            add_info = ''
            prompt = self.prepare_prompt(agent=focal_agent, player_type=player_type, additional_info=add_info)
            tasks.append({
                'index': i,
                'agent': focal_agent,
                'prompt': prompt,
                'player_type': player_type
            })

        # 使用线程池并行执行所有任务
        max_workers = min(self.num_agents, 10)  # 限制最大并发数为10，避免API限流
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"{player_type}_worker") as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self._call_llm, task['agent'], task['prompt'], task['player_type']): task['index']
                for task in tasks
            }

            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    focal_agent_response = future.result(timeout=60)  # 60秒超时
                    results[index] = focal_agent_response
                except Exception as e:
                    logger.error(f"❌ Agent {tasks[index]['agent'].unique_id} 调用失败: {str(e)}")
                    # 失败时使用默认值
                    results[index] = None

        # 按顺序处理结果
        for i in range(self.num_agents):
            focal_agent = self.get_agent(i)
            focal_agent_response = results.get(i)
            agent_key = f"agent_{focal_agent.unique_id}_{focal_agent.type_restriction}"
            if agent_key not in self.dialogs[round_key]:
                self.dialogs[round_key][agent_key] = {}

            if focal_agent_response is None:
                logger.warning(f"⚠️ Agent {focal_agent.unique_id} 响应为空，使用默认决策")
                send_amounts = {}
                for neighbor_id in focal_agent.neighbor_ids:
                    send_amounts[neighbor_id] = 0

                content_str = "ERROR"
                reasoning_str = "ERROR"
            else:
                send_amounts = self.extract_decisions_regex(focal_agent_response.content if hasattr(focal_agent_response, 'content') else str(focal_agent_response))
                content_str = str(focal_agent_response.content)
                reasoning_str = str(focal_agent_response.reasoning_content) if hasattr(focal_agent_response, 'reasoning_content') else ""

            if player_type == "investor":
                logger.info(f"📝 investor agent_{focal_agent.unique_id} 的决策为: {send_amounts}")
                focal_agent.I_invested_1.append(send_amounts)
                self.agents_invested_amounts[focal_agent.unique_id] = send_amounts
            elif player_type == "trustee":
                logger.info(f"📝 trustee agent_{focal_agent.unique_id} 的决策为: {send_amounts}")
                focal_agent.T_returned_3.append(send_amounts)
                self.agents_returned_amounts[focal_agent.unique_id] = send_amounts

            self.dialogs[round_key][agent_key][f"as_{player_type}"] = {
                "prompt": tasks[i]['prompt'],
                "content": content_str,
                "reasoning_content": reasoning_str
            }

        logger.info(f"✅ 实验共{self.params.rounds}轮，第 {self.step} 轮-{player_type} 阶段完成！\n")

    def prepare_prompt(self, agent, player_type, additional_info="") -> str:

        prompt = additional_info
        if self.step == 1:
            if player_type == "investor":
                decision_stages = p_decision_stages.get("investor").format(step=self.step, type=agent.type_restriction)
                prompt = decision_stages + p_end

            elif player_type == "trustee":
                received_amounts = agent.T_received_2[-1]
                investor_decision = p_decision.format(step=self.step, received_amounts=received_amounts)
                decision_stages = p_decision_stages.get("trustee")
                prompt = investor_decision + decision_stages + p_end

        else:
            if player_type == "investor":
                memory = p_memory.format(I_invested_1=agent.I_invested_1[-1], I_received_4=agent.I_received_4[-1], T_received_2=agent.T_received_2[-1], T_returned_3=agent.T_returned_3[-1], payoff=agent.payoff)
                decision_stages = p_decision_stages.get("investor").format(step=self.step, type=agent.type_restriction)
                prompt = memory + decision_stages + p_end

            elif player_type == "trustee":
                received_amounts = agent.T_received_2[-1]
                investor_decision = p_decision.format(step=self.step, received_amounts=received_amounts)
                decision_stages = p_decision_stages.get("trustee")
                prompt = investor_decision + decision_stages + p_end

        return prompt

    def _call_llm(self, focal_agent, prompt, player_type="Investor"):
        import time
        start_time = time.time()
        try:
            focal_agent_response = focal_agent.llm_agent.step(prompt).msgs[0]

            # Debug
            # invest_value = {}
            # for id in focal_agent.neighbor_ids:
            #     invest_value[id] = random.randint(1, 5)
            # focal_agent_response = f"\nThis round, I decide to send {invest_value} to each neighbor."

            # elapsed_time = time.time() - start_time
            # logger.info(f"✅ [{player_type}] Agent {focal_agent.unique_id} 投资决策完成 - API耗时: {elapsed_time:.2f}秒")
            # logger.debug(f"  响应: {str(focal_agent_response)[:100]}...")

            return focal_agent_response
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"❌ [{player_type}] Agent {focal_agent.unique_id} 调用失败 - 耗时: {elapsed_time:.2f}秒 - 错误: {str(e)}")
            raise

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

            logger.debug(f"Agent {i} 收到邻居返还: {returned_from_neighbors}")

    def extract_decisions_regex(self, answer):
        pattern = r"This round,\s*I\s+decide\s+to\s+send\s*(?:\[([^\]]+)\]|\{([^}]+)\})"
        match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)

        if not match:
            logger.warning(f"未找到标准格式，尝试备用提取方式。原始响应: {answer[:100]}")
            return self._extract_fallback(answer)

        content = match.group(1) if match.group(1) else match.group(2)

        pairs = re.findall(r"(\d+)\s*:\s*(\d+)", content)

        if not pairs:
            logger.warning(f"找到格式但未解析到键值对: {content}")
            return {}

        result = {}
        for neighbor_id, amount in pairs:
            result[int(neighbor_id)] = min(int(amount), 5)

        logger.debug(f"提取发送金额决策决策: {result}")
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

    def run_model(self, max_round=50) -> tuple[list, dict]:
        """运行模型指定步数"""
        import time
        import tqdm

        logger.info(f"\n{'=' * 70}")
        logger.info(f"🚀 开始运行信任博弈实验")
        logger.info(f"{'=' * 70}")
        logger.info(f"  • 模型类型: {self.model_type}")
        logger.info(f"  • 博弈类型: {self.game_type}")
        logger.info(f"  • 智能体数量: {self.num_agents}")
        logger.info(f"  • 网格大小: {self.width} x {self.height}")
        logger.info(f"  • 自由玩家比例: {self.proportion}")
        logger.info(f"  • 运行轮次: {max_round}")
        logger.info(f"  • Run ID: {self.run_id}")
        logger.info(f"  • Iteration: {self.iteration}")
        logger.info(f"{'=' * 70}\n")

        start_time = time.time()

        for _ in tqdm.trange(max_round):
            self._step()

        elapsed_time = time.time() - start_time

        # 计算统计信息
        total_interactions = len(self.all_data)
        avg_invested = sum(row['invested_amount'] for row in self.all_data) / total_interactions if total_interactions > 0 else 0
        avg_returned = sum(row['returned_amount'] for row in self.all_data) / total_interactions if total_interactions > 0 else 0
        avg_investor_payoff = sum(row['investor_agent_payoff'] for row in self.all_data) / total_interactions if total_interactions > 0 else 0
        avg_trustee_payoff = sum(row['trustee_agent_payoff'] for row in self.all_data) / total_interactions if total_interactions > 0 else 0

        # 按智能体类型统计
        free_agents_data = [row for row in self.all_data if row['investor_agent_type'] == 'free']
        constrained_agents_data = [row for row in self.all_data if row['investor_agent_type'] == 'constrained']

        avg_invested_free = sum(row['invested_amount'] for row in free_agents_data) / len(free_agents_data) if free_agents_data else 0
        avg_invested_constrained = sum(row['invested_amount'] for row in constrained_agents_data) / len(constrained_agents_data) if constrained_agents_data else 0

        all_dialogue = {"initial_sys_prompt": self.initial_sys_prompt,
                        "dialogs": self.dialogs,
                        "agent_invested_amounts": self.agents_invested_amounts,
                        "agent_returned_amounts": self.agents_returned_amounts, }

        # 打印最终统计
        logger.info(f"\n{'=' * 70}")
        logger.info(f"🎉 实验运行完成！")
        logger.info(f"{'=' * 70}")
        logger.info(f"📊 核心指标统计:")
        logger.info(f"  • 总交互次数: {total_interactions}")
        logger.info(f"  • 平均委托金额: {avg_invested:.3f} / 5.0")
        logger.info(f"  • 平均返还金额: {avg_returned:.3f}")
        logger.info(f"  • 投资者平均收益: {avg_investor_payoff:.3f}")
        logger.info(f"  • 受托者平均收益: {avg_trustee_payoff:.3f}")
        logger.info(f"  • 总体平均收益: {(avg_investor_payoff + avg_trustee_payoff):.3f}")
        logger.info(f"\n👥 分类型统计:")
        logger.info(f"  • 自由玩家平均委托: {avg_invested_free:.3f}")
        logger.info(f"  • 受限玩家平均委托: {avg_invested_constrained:.3f}")
        logger.info(f"  • 差异: {abs(avg_invested_free - avg_invested_constrained):.3f}")
        logger.info(f"\n⏱️ 性能统计:")
        logger.info(f"  • 总耗时: {elapsed_time:.2f}秒")
        logger.info(f"  • 平均每轮耗时: {elapsed_time / max_round:.2f}秒")
        logger.info(f"  • 平均每交互耗时: {elapsed_time / total_interactions * 1000:.2f}毫秒")
        logger.info(f"\n💾 数据记录:")
        logger.info(f"  • 对话轮次记录: {len(self.dialogs)} 轮")
        logger.info(f"  • 数据行数: {len(self.all_data)}")
        logger.info(f"{'=' * 70}\n")

        return self.all_data, all_dialogue

    def get_agent(self, agent_id):
        """根据ID获取智能体"""
        return self.agents[agent_id]

    def _collect_data(self):
        """
        收集当前轮次所有两两博弈的数据（每个方向一条记录）。
        每条记录对应一次投资者-受托者博弈。
        """
        for agent in self.agents:
            # 获取当前轮次的数据
            invested_dict = agent.I_invested_1[-1] if agent.I_invested_1 else {}  # 作为投资者，委托给各邻居的金额
            returned_dict = agent.I_received_4[-1] if agent.I_received_4 else {}  # 作为投资者，从各邻居收到的返还金额

            for neighbor_id in agent.neighbor_ids:
                invested_amount = invested_dict.get(neighbor_id, 0)
                returned_amount = returned_dict.get(neighbor_id, 0)

                # 计算收益
                investor_payoff = (5 - invested_amount) + returned_amount
                trustee_payoff = 3 * invested_amount - returned_amount

                row = {
                    "run_id": self.run_id,
                    "iteration": self.iteration,
                    "round": self.step,
                    "num_agents": self.num_agents,
                    "proportion": self.proportion,
                    "investor_agent_id": agent.unique_id,
                    "investor_agent_type": agent.type_restriction,
                    "trustee_neighbor_id": neighbor_id,
                    "invested_amount": invested_amount,
                    "returned_amount": returned_amount,
                    "investor_agent_payoff": investor_payoff,
                    "trustee_agent_payoff": trustee_payoff,
                }
                self.all_data.append(row)
    def reset_record(self):
        """重置记录"""
        pass