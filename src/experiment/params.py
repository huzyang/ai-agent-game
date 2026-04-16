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

from mesa.experimental.scenarios import Scenario
class GameScenario(Scenario):
    """Scenario for model."""
    num_agents: int = 4  # 节点数
    width: int = int(num_agents ** 0.5)  # 根号 N
    height: int = width
    torus: bool = True
    model_type: str = ModelType.QWEN3_5_FLASH.value
    game_type: str = GameType.TRUST.value
    proportion: float = 0.5 # 自由节点比例

class Params:

    def __init__(self):
        self.num_agents = GameScenario.num_agents  # 节点数
        self.width: int = int(self.num_agents ** 0.5)  # 根号 N
        self.height: int = self.width
        self.proportion = GameScenario.proportion  # 自由节点比例
        self.model_type = GameScenario.model_type
        self.game_type = GameScenario.game_type
        self.rounds = 2  # 游戏轮数
        self.iterations = 1

        ################# LLM 参数 ####################
        load_dotenv()
        self.api_key = os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("请设置环境变量 QWEN_API_KEY")
        self.api_base_url = os.getenv("QWEN_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model_type_list = [ModelType.QWEN3_5_FLASH]
        self.temperature = 0.5
        self.max_tokens = 4096
        ################# mesa 参数 ####################
        self.number_processes = 4
        self.data_collection_period = 1

        ################# 保存信息 #####################
        self.output_dir = os.path.join(CommonUtils.get_project_root_path(), "outputs")

    @property
    def model_init_params(self):
        return {
            'num_agents': GameScenario.num_agents,
            'width': GameScenario.width,
            'height': GameScenario.height,
            'torus': GameScenario.torus,
            'model_type': GameScenario.model_type,
            'game_type': GameScenario.game_type,
            'proportion': GameScenario.proportion,
            'rounds': self.rounds
        }

    def print_model_init_params(self):
        """打印模型初始化参数"""
        print(f"\n模型初始化参数:")
        for key, value in self.model_init_params.items():
            print(f"  {key} = {value}")

    def print_all_params(self):
        """打印所有参数"""
        pass

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
