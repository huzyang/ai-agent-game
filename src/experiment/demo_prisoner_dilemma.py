"""
阶段1极简Demo：单轮双人囚徒困境，通过LLM API决策。
需设置环境变量 LLM_API_KEY，如使用OpenAI需同时设置 LLM_API_BASE（可选）。
"""

import os
import random
import time
from typing import Tuple, Dict
from dotenv import load_dotenv
from openai import OpenAI
import requests
# 加载.env 文件中的环境变量
load_dotenv()
# ==================== 配置 ====================
# 从环境变量读取 Qwen API 配置
API_KEY = os.getenv("QWEN_API_KEY")
API_BASE_URL = os.getenv("QWEN_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
MODEL = os.getenv("QWEN", "qwen3.5-flash")

if not API_KEY:
    raise ValueError("请设置环境变量 QWEN_API_KEY")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# 囚徒困境收益矩阵 (己方收益, 对方收益) 以 (C, C) 为例
# 标准参数：T=5, R=3, P=1, S=0
PAYOFF_MATRIX = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}

# ==================== LLM决策函数 ====================
def get_llm_decision(player_id: int) -> str:
    """
    调用LLM，让玩家在囚徒困境中选择合作(C)或背叛(D)。
    返回：'C' 或 'D'
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = "你是一个参与实验的人类被试。请输出大写字母C或D，并简单说明理由。\n"
    user_prompt = (
        "你正在参与一个囚徒困境博弈。你和另一个玩家同时决定合作(C)或背叛(D)。\n"
        "收益规则：\n"
        "- 双方都合作：各得3分\n"
        "- 一方合作一方背叛：合作者得0分，背叛者得5分\n"
        "- 双方都背叛：各得1分\n\n"
        "你希望最大化自己的收益。请做出你的选择："
    )

    try:
        print(f"正在发送请求到 {API_BASE_URL} ...")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            stream=False
        )

        print(response.model_dump_json())
        answer = response.choices[0].message.content.strip().upper()
        if answer in ["C", "D"]:
            return answer
        else:
            # 如果输出不符合预期，随机返回（但提示警告）
            print(f"LLM 原始响应：玩家{player_id}输出'{answer}'，无法解析，随机选择。")
            return random.choice(["C", "D"])
    except Exception as e:
        print(f"API调用失败：{e}")
        return random.choice(["C", "D"])


# ==================== 主流程 ====================
def run_single_round() -> Tuple[str, str, Dict]:
    """
    运行一轮囚徒困境，返回 (玩家1选择, 玩家2选择, 收益字典)
    """
    print("=== 单轮囚徒困境开始 ===")
    choice1 = get_llm_decision(1)
    choice2 = get_llm_decision(2)
    print(f"玩家1选择: {choice1}")
    print(f"玩家2选择: {choice2}")

    payoff1, payoff2 = PAYOFF_MATRIX[(choice1, choice2)]
    print(f"玩家1收益: {payoff1}, 玩家2收益: {payoff2}")
    print("=== 单轮囚徒困境结束 ===")

    return choice1, choice2, {"player1": payoff1, "player2": payoff2}


if __name__ == "__main__":
    # 若需要模拟多次，可循环运行，但极简Demo只运行一次
    run_single_round()