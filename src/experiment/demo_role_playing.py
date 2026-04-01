from colorama import Fore
from camel.agents import ChatAgent
from camel.societies import RolePlaying
from camel.utils import print_text_animated
from camel.models import ModelFactory
from camel.types import ModelPlatformType
import os

from dotenv import load_dotenv
load_dotenv()

# ==================== 配置 ====================
# 从环境变量读取 Qwen API 配置
API_KEY = os.getenv("QWEN_API_KEY")
API_BASE_URL = os.getenv("QWEN_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
MODEL = os.getenv("QWEN", "qwen3.5-flash")

if not API_KEY:
    raise ValueError("请设置环境变量 QWEN_API_KEY")

# 囚徒困境收益矩阵 (己方收益, 对方收益) 以 (C, C) 为例
# 标准参数：T=5, R=3, P=1, S=0
PAYOFF_MATRIX = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type=MODEL,
    url=API_BASE_URL,
    api_key=API_KEY,
    model_config_dict={"temperature": 1, "max_tokens": 20},
)

def pd_game(model=model, chat_turn_limit=2):
    system_prompt = "你是一个参与经济行为实验的人类。\n"

    task_prompt = (
        "你是一个参与经济行为实验的人类。\n"
        "你正在参与一个囚徒困境博弈。你和另一个玩家同时决定合作(C)或背叛(D)。\n"
        "收益规则：\n"
        "- 双方都合作：各得3分\n"
        "- 一方合作一方背叛：合作者得0分，背叛者得5分\n"
        "- 双方都背叛：各得1分\n\n"
        "你希望最大化自己的收益。你无须输出其他内容，只需要回答选择合作(C)或背叛(D)。"
    ) #设置任务目标
    role_play_session = RolePlaying(
        user_role_name="实验人员",#设置用户角色名，在roleplay中，user用于指导AI助手完成任务
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=False,
        task_specify_agent_kwargs=dict(model=model),
        output_language='中文'#设置输出语言
    )
    print(
        Fore.BLUE + f"AI 用户系统消息:\n{role_play_session.user_sys_msg}\n"
    )

    print(Fore.YELLOW + f"原始任务提示:\n{task_prompt}\n")

    n = 0
    input_msg = role_play_session.init_chat()
    while n < chat_turn_limit:
        n += 1
        user_response = role_play_session.step(input_msg)
        if user_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI 用户已终止。"
                    f"原因: {user_response.info['termination_reasons']}."
                )
            )
            break

        print_text_animated(
            Fore.BLUE + f"AI 用户:\n\n{user_response.msg.content}\n"
        )

        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break

    return role_play_session

def main(model=model, chat_turn_limit=2) -> None:
    print("=== 单轮囚徒困境开始 ===")
    # role_play_session = pd_game(model=model, chat_turn_limit=chat_turn_limit)

    task_prompt = (
        "你是一个参与经济行为实验的人类。\n"
        "你正在参与一个囚徒困境博弈。你和另一个玩家同时决定合作(C)或背叛(D)。\n"
        "收益规则：\n"
        "- 双方都合作：各得3分\n"
        "- 一方合作一方背叛：合作者得0分，背叛者得5分\n"
        "- 双方都背叛：各得1分\n\n"
        "你希望最大化自己的收益。你先用一句话概括你的目的，然后回答选择合作(C)或背叛(D)。"
    ) #设置任务目标
    agent = ChatAgent(
        model=model,
        output_language='中文'
    )

    response = agent.step(task_prompt)
    print(response.msgs[0].content)


if __name__ == "__main__":
    main()
