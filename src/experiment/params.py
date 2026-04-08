"""
实验参数
"""
from enum import Enum
from src.utils import CommonUtils
from dotenv import load_dotenv
import os

class ModelType(Enum):
    GPT_4 = "gpt-4"
    QWEN3_5_FLASH = "qwen3.5-flash"
    QWEN3_5_MAX = "qwen3.5-max"

    @property
    def is_openai(self) -> bool:
        r"""Returns whether this type of models is an OpenAI-released model."""
        return self in {
            ModelType.GPT_4,
        }

    @property
    def token_limit(self) -> int:
        r"""Returns the maximum token limit for a given model.
        Returns:
            int: The maximum token limit for the given model.
        """
        if self is ModelType.GPT_4:
            return 8192
        elif self is ModelType.QWEN3_5_FLASH:
            return 4096
        else:
            raise ValueError("Unknown model type")


class GameType(Enum):
    PDG = "pd_game"
    TRUST = "trust_game"

class Params:

    def __init__(self):
        self.N = 4  # 节点数
        self.width: int = int(self.N ** 0.5)  # 根号 N
        self.height: int = self.width
        self.p = 0.5  # 自由节点比例
        self.rounds = 2  # 游戏轮数
        self.game_type = GameType.TRUST.value

        ################# LLM 参数 ####################
        load_dotenv()
        self.api_key = os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("请设置环境变量 QWEN_API_KEY")
        self.api_base_url = os.getenv("QWEN_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model_list = [ModelType.QWEN3_5_FLASH]
        self.temperature = 0.7
        self.max_tokens = 4096
        ################# mesa 参数 ####################
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
        print(f"\n对称信任博弈实验参数:")
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
