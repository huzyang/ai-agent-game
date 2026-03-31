import random
import mesa
import numpy as np
from enum import IntEnum


class StrategyType(IntEnum):
    """离散策略类型枚举 - 使用整数表示"""
    IT = 0  # 投资 + 可信
    IU = 1  # 投资 + 不可信
    NT = 2  # 不投资 + 可信
    NU = 3  # 不投资 + 不可信

    @classmethod
    def get_strategy_name(cls, value):
        """获取策略名称"""
        names = {
            0: "IT",
            1: "IU",
            2: "NT",
            3: "NU"
        }
        return names.get(value, f"Unknown({value})")

    @classmethod
    def from_string(cls, strategy_str):
        """从字符串转换"""
        strategy_map = {
            "IT": cls.IT,
            "IU": cls.IU,
            "NT": cls.NT,
            "NU": cls.NU
        }
        return strategy_map.get(strategy_str, cls.IT)

    @classmethod
    def to_strategy(cls, investor_strategy, truster_strategy):
        """根据投资策略和信任策略生成策略"""
        if investor_strategy == "I":
            if truster_strategy == "T":
                return StrategyType.IT
            else:
                return StrategyType.IT
        else:
            if truster_strategy == "T":
                return StrategyType.NT
            else:
                return StrategyType.NU


class TrustAgent(mesa.Agent):
    """信任博弈智能体 - 对称信任博弈版本"""

    def __init__(self, unique_id, model, initial_strategy):
        """
        初始化智能体

        参数:
        - unique_id: 唯一标识符
        - model: 所属模型
        - initial_strategy: 初始策略，如果为None则按模型分布分配
        """
        super().__init__(model)
        self.unique_id = unique_id

        self.strategy = initial_strategy
        self.next_strategy = self.strategy
        self.payoff = 0.0  # 每一轮的最终收益

        # 邻居信息 - 修改存储格式
        self.neighbors_1hop = []  # 邻居id

        # 策略历史
        self.strategy_history = []

    def __str__(self):
        return (f"Agent {self.unique_id}: Strategy: (now: {StrategyType.get_strategy_name(self.strategy)}, next: {StrategyType.get_strategy_name(self.next_strategy)}), Payoff: {self.payoff:.2f}, Pairwise payoff: (avg: {self.avg_pairwise_games_payoff}, total: {self.total_pairwise_games_payoff}), "
                f"Group payoff: (avg: {self.avg_group_games_payoff}, total: {self.total_group_games_payoff})")

    def update_payoff(self):
        """
        不归一化收益，按比例相加
        """
        beta = self.model.beta
        self.payoff = round((1 - beta) * self.total_pairwise_games_payoff + (beta * self.total_group_games_payoff), 4)
        # 更新全局收益
        self.model.agents_payoff[self.unique_id] = self.payoff

    def reset_counts(self):
        """重置计数和当前收益"""
        self.payoff = 0.0

    def _imitation(self):
        """
        使用比例模仿规则更新策略

        Returns:
            bool: 如果智能体采用了邻居的策略则返回True，否则返回False
        """
        # 获取邻居节点 - 修正为使用self.neighbors_1hop，与_fermi_update保持一致
        if not self.neighbors_1hop:
            return False

        # 随机选择一个邻居
        neighbor_id = np.random.choice(self.neighbors_1hop)
        neighbor = self.model.get_agent(neighbor_id)

        if neighbor is None:
            return False

        # 获取邻居和当前智能体的收益
        neigh_payoff = neighbor.payoff
        focal_agent_payoff = self.payoff

        # 最大收益差
        phi = 61

        if neigh_payoff > focal_agent_payoff:
            # 计算采用邻居策略的概率
            prob = min(1.0, (neigh_payoff - focal_agent_payoff) / phi)  # 限制概率不超过1
            # 生成一个 0 到 1 之间的随机数
            r = np.random.random()

            # 记录调试信息（可选）
            # print(
            #     f"Focal agent ID: {self.unique_id}, "
            #     f"neighbor payoff is {neigh_payoff} and focal is {focal_agent_payoff}, "
            #     f"prob is {round(prob,5)} and r is {round(r,5)}"
            # )

            # 检查智能体是否采用邻居的策略
            if r <= prob:
                # 更改策略
                self.next_strategy = neighbor.strategy
                return True

        return False

    def choose_next_strategy(self):
        """
        选择下一个策略（用于同步更新）,但不应用新策略
        更新next_strategy
        """
        match self.model.update_rule:
            case "Fermi":
                return self._fermi_update()
            case "Imitation":
                return self._imitation()
            case "Higher-order-Imitation":
                return self._higher_order_imitation()
            case "Regret-Minimization":
                return self._regret_minimization_update()
            case "QL":
                pass
            case _:
                raise ValueError(f"Unknown update rule !")

        return False

    def apply_next_strategy(self):
        """
        应用下一个策略（用于同步更新）
        """
        if self.next_strategy != self.strategy:
            # 记录更新信息
            self.strategy_history.append({
                'step': self.model.steps,
                'old_strategy': self.strategy,
                'new_strategy': self.next_strategy
            })
            self.strategy = self.next_strategy  # 更新策略
            # 更新全局策略 - 确保在策略改变时立即更新
            self.model.agents_strategy[self.unique_id] = self.strategy
            return True
        return False

    def step(self):
        pass

    def advance(self):
        pass
