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

        ################# mesa 模型参数 ####################
        self.iterations = 1
        self.number_processes = 4
        self.data_collection_period = 1

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

        print(f"{'=' * 50}\n")

    def record_params(self):
        """将参数转换为字典"""
        str_result = ""
        str_result += f"\n    \"N\" =  {self.N} "
        str_result += f"\n    \"p\" =  {self.p} "
        str_result += f"\n    \"rounds\" =  {self.rounds} "


        return str_result
    def to_dict(self):
        """将参数转换为字典"""
        return self.__dict__