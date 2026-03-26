# params.py - 对称信任博弈参数管理
from enum import Enum
from src.utils import CommonUtils

__all__ = ["Params"]


class ANALYZED_PARAMS(Enum):
    R = "r" # 信任困境强度
    P = "p" # 标记为二单形的三角形比例
    X = "x" # 投资比例
    Y = "y" # 可信受托者返还比例
    R_T = "r_t" # 可信受托者倍增因子
    DELTA = "delta" # 高阶模仿规则参数

class UPDATE_RULE_TYPE(Enum):
    Q_LEARNING = "Q-Learning"
    FERMI = "Fermi"
    IM = "Imitation"
    HOIM = "Higher-order-Imitation"
    REGRET_MIN = "Regret-Minimization"

class Params:
    """对称信任博弈参数类"""

    def __init__(self):
        # 模型初始化参数
        self.N = 400  # 节点数
        self.network_file = CommonUtils.get_project_root_path() + f"/datas/hexagonal_lattice_{self.N}.csv"  # 网络参数
        self.initial_investors_ratio = 0.5  # 初始投资者比例
        self.initial_prosocial_ratio = 0.5  # 初始亲社会比例
        self.x = 1.0  # 投资比例
        # self.x = [0.2, 0.4, 0.6, 0.8, 1.0]  # 投资比例
        self.y = 0.5  # 返还比例
        # self.y = [0.1, 0.3, 0.5, 0.7, 0.9]  # 返还比例
        self.r_t = 3.0  # 可信倍增因子
        # self.r_t = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # 可信倍增因子
        self.r = 0.3  # 信任困境强度
        # self.r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 信任困境强度
        # self.p = 1  # 标记为二单形的三角形比例
        self.p = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.9]  # 标记为二单形的三角形比例
        # self.delta = 0.1  # 高阶模仿规则参数
        self.delta = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,]  # 高阶模仿规则参数

        # 演化参数
        self.update_rule = UPDATE_RULE_TYPE.HOIM.value  # 更新规则
        self.is_normalized = True  # 最终payoff统计是否归一化
        self.fermi_kappa = 50  # 若用Fermi，噪声因子必须远远大于1
        self.epsilon = 0.02  # Q-learning相关参数
        self.update_interval = 1  # 更新间隔
        self.activation_order = 'Simultaneous'  # 激活顺序

        # 数据收集参数
        self.analyzed_params = [ANALYZED_PARAMS.DELTA.value, ANALYZED_PARAMS.P.value]
        self.collected_datas = ["IT", "IU", "NT", "NU", "GW"] #, "FI", "FT"]

        # 批量运行设置
        self.iterations = 20
        self.max_steps = 6000
        self.number_processes = 8
        self.data_collection_period = 1

    @property
    def model_init_params(self):
        return {
            'N': self.N,
            'initial_investors_ratio': self.initial_investors_ratio,
            'initial_prosocial_ratio': self.initial_prosocial_ratio,
            'x': self.x,
            'y': self.y,
            'r_t': self.r_t,
            'r': self.r,
            'p': self.p,
            'delta': self.delta
        }

    def print_model_init_params(self):
        """打印模型初始化参数"""
        print(f"\n模型初始化参数:")
        for key, value in self.model_init_params.items():
            print(f"  {key} = {value}")

    def print_all_params(self):
        """打印所有参数"""
        print(f"\n{'=' * 50}")
        print(f"对称信任博弈参数")
        print(f"{'=' * 50}")

        print(f"\n模型初始化参数:")
        print(f"  节点数量 N = {self.N}")
        print(f"  初始投资者比例 initial_investors_ratio = {self.initial_investors_ratio}")
        print(f"  初始亲社会比例 initial_prosocial_ratio = {self.initial_prosocial_ratio}")
        print(f"  投资比例 x = {self.x}")
        print(f"  返还比例 y = {self.y}")
        print(f"  可信受托人倍增因子 r_t = {self.r_t}")
        print(f"  信任困境强度 r = {self.r}")
        print(f"  标记为二单形的三角形比例 p = {self.p}")
        print(f"  高阶模仿规则参数 delta = {self.delta}")

        print(f"\n网络参数:")
        print(f"  网络配置 network_file = {self.network_file}")

        print(f"\n演化参数:")
        print(f"  fermi_kappa = {self.fermi_kappa}")
        print(f"  epsilon = {self.epsilon}")
        print(f"  update_rule = {self.update_rule}")
        print(f"  update_interval = {self.update_interval}")
        print(f"  activation_order = {self.activation_order}")
        print(f"  is_normalized = {self.is_normalized}")

        print(f"\n数据收集参数:")
        print(f"  analyzed_params = {self.analyzed_params}")
        print(f"  collected_datas = {self.collected_datas}")

        print(f"\n批量运行设置:")
        print(f"  iterations = {self.iterations}")
        print(f"  max_steps = {self.max_steps}")
        print(f"  number_processes = {self.number_processes}")
        print(f"  data_collection_period = {self.data_collection_period}")

        print(f"{'=' * 50}\n")

    def record_params(self):
        """将参数转换为字典"""
        str_result = ""
        str_result += f"\n    \"N\" =  {self.N} "
        str_result += f"\n    \"network_file\" = \" {self.network_file}\" "
        str_result += f"\n    \"initial_investors_ratio\" =  {self.initial_investors_ratio} "
        str_result += f"\n    \"initial_prosocial_ratio\" =  {self.initial_prosocial_ratio} "
        str_result += f"\n    \"x\" =  {self.x} "
        str_result += f"\n    \"y\" =  {self.y} "
        str_result += f"\n    \"r_t\" =  {self.r_t} "
        str_result += f"\n    \"r\" =  {self.r} "
        str_result += f"\n    \"p\" =  {self.p} "
        str_result += f"\n    \"delta\" =  {self.delta} "
        str_result += f"\n\n    \"update_rule\" = \" {self.update_rule}\" "
        str_result += f"\n    \"fermi_kappa\" =  {self.fermi_kappa} "
        str_result += f"\n    \"epsilon\" =  {self.epsilon} "
        str_result += f"\n    \"update_interval\" =  {self.update_interval} "
        str_result += f"\n    \"activation_order\" = \" {self.activation_order}\" "
        str_result += f"\n    \"is_normalized\" =  {self.is_normalized} "
        str_result += f"\n\n    \"analyzed_params\" =  {self.analyzed_params} "
        str_result += f"\n    \"collected_datas\" =  {self.collected_datas} "
        str_result += f"\n\n    \"iterations\" =  {self.iterations} "
        str_result += f"\n    \"max_steps\" =  {self.max_steps} "
        str_result += f"\n    \"number_processes\" =  {self.number_processes} "
        str_result += f"\n    \"data_collection_period\" =  {self.data_collection_period} "

        return str_result
    def to_dict(self):
        """将参数转换为字典"""
        return self.__dict__