"""
智能体基类：存储历史、类型、邻居列表，定义决策接口。
"""

from typing import List, Dict, Any, Optional
import random
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("QWEN_API_KEY")
API_BASE_URL = os.getenv("QWEN_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("QWEN", "qwen3.5-flash")

if not API_KEY:
    raise ValueError("请设置环境变量 QWEN_API_KEY")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


class BaseAgent:
    """智能体基类"""
    def __init__(self, agent_id: int, agent_type: str, neighbors: List[int]):
        self.id = agent_id
        self.type = agent_type  # "constrained" 或 "free"
        self.neighbors = neighbors  # 邻居ID列表
        self.history = []  # 存储每轮决策与收益，格式：{round: int, decisions: dict, payoffs: dict}

    def add_history(self, round_num: int, decisions: Dict[int, Any], payoffs: Dict[int, float]):
        """记录一轮历史"""
        self.history.append({
            "round": round_num,
            "decisions": decisions,  # {neighbor_id: decision}
            "payoffs": payoffs       # {neighbor_id: payoff}
        })

    def get_last_round(self) -> Optional[Dict]:
        """获取上一轮历史"""
        return self.history[-1] if self.history else None

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用LLM获取原始响应"""
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=50,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Agent {self.id} LLM调用失败: {e}")
            return ""

    def decide(self, game_name: str, context: Dict[str, Any]) -> Dict[int, Any]:
        """
        根据游戏类型和上下文做出决策。
        子类需重写，返回 {neighbor_id: decision}。
        """
        raise NotImplementedError