# params.py - 对称信任博弈参数管理
from enum import Enum
from src.utils import CommonUtils

__all__ = ["Params"]


class MODEL_TYPE(Enum):
    QWEN3_5_FLASH = "qwen3.5-flash"
    QWEN3_5_MAX = "qwen3.5-max"

class Params:

    def __init__(self):
        self.N = 9  # 节点数
        self.p = 0.5  # 自由节点比例

        self.rounds = 2  # 游戏轮数

        ################# 暂时不用的参数 ####################
        self.network_file = CommonUtils.get_project_root_path() + f"/datas/hexagonal_lattice_{self.N}.csv"  # 网络参数
        self.p = 0.5  # 自由节点比例

        # 演化参数
        self.is_normalized = True  # 最终payoff统计是否归一化
        self.update_interval = 1  # 更新间隔

        # 数据收集参数
        self.analyzed_params = [ANALYZED_PARAMS.DELTA.value, ANALYZED_PARAMS.P.value]
        self.collected_datas = ["IT", "IU", "NT", "NU", "GW"] #, "FI", "FT"]

    @property
    def model_init_params(self):
        return {
            'N': self.N,
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
        print(f"  标记为自由节点的比例 p = {self.p}")
        print(f"  游戏轮数 rounds = {self.rounds}")

        print(f"\n网络参数:")
        print(f"  网络配置 network_file = {self.network_file}")

        print(f"\n演化参数:")
        print(f"  is_normalized = {self.is_normalized}")

        print(f"\n数据收集参数:")
        print(f"  analyzed_params = {self.analyzed_params}")
        print(f"  collected_datas = {self.collected_datas}")

        print(f"{'=' * 50}\n")

    def record_params(self):
        """将参数转换为字典"""
        str_result = ""
        str_result += f"\n    \"N\" =  {self.N} "
        str_result += f"\n    \"p\" =  {self.p} "
        str_result += f"\n    \"rounds\" =  {self.rounds} "

        str_result += f"\n    \"network_file\" = \" {self.network_file}\" "
        str_result += f"\n    \"update_interval\" =  {self.update_interval} "
        str_result += f"\n    \"is_normalized\" =  {self.is_normalized} "
        str_result += f"\n\n    \"analyzed_params\" =  {self.analyzed_params} "
        str_result += f"\n    \"collected_datas\" =  {self.collected_datas} "

        return str_result
    def to_dict(self):
        """将参数转换为字典"""
        return self.__dict__