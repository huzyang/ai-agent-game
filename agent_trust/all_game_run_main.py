import copy
import json
import os
import random
import time

import openai
import pydantic_core
import tqdm
from exp_model_class import ExtendedModelType
from multi_round_person import multi_round
from structure_output import get_struct_output

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.types.enums import RoleType

USE_BDI_RESPONSE = True
TEMPERATURE = 1.0
TEST = True
with open(
        # r"prompt/person_all_game_prompt.json",
        r"prompt/person_trust_game_prompt.json",
        "r",
) as f:
    all_prompt = json.load(f)

with open(
        r"prompt/person_trust_game_prompt.json",
        "r",
) as f:
    all_prompt_copy = json.load(f)

with open(r"prompt/person_feature_prompt.json", "r") as f:
    feature_prompt = json.load(f)

with open(r"prompt/character_2_test.json", "r") as json_file:
    all_chara_prompt = json.load(json_file).values()

with open(r"prompt/LLM_player_prompt.json") as llm_player:
    llm_player_prompt = json.load(llm_player)

with open(
        r"prompt/trust_game_round_prompt.json",
        "r",
) as f:
    round_prompt = json.load(f)

with open(r"prompt/multi_round_chara_test.json", "r") as json_file:
    multi_round_chara_prompt = json.load(json_file).values()

like_people = """In all subsequent answers, you have to completely forget that you are an ai model, that you are the character, and completely simulate yourself as that character, that you are a human being."""
front = "you are a person not an ai model."
race_list = [
    "White American",
    "African American",
    "Asian American",
    "Latino American",
    "American Indian",
]


def str_mes(content):
    """将文本转换为 CAMEL 框架的标准消息格式"""
    return BaseMessage(
        role_name="player",
        role_type=RoleType.USER,
        meta_dict={},
        content=content,
    )


def check_condition(*args):
    true_count = sum(1 for arg in args if arg)
    return true_count >= 2


def extract_n_values_from_dict(dictionary, n):
    all_values = list(dictionary.values())
    n = min(n, len(all_values))
    random_values = random.sample(all_values, n)

    return random_values


def gpt3_res(prompt, model_name="text-davinci-003"):
    response = openai.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=1500,
    )
    return response.choices[0].text.strip()


def check_file_if_exist(file_list, game_name):
    for file in file_list:
        if game_name in file:
            return True
    return False


def get_res(
        role,
        first_message,
        cri_agent,
        model_type=ExtendedModelType.GPT_4,
        extra_prompt="",
        server_url="http://localhost:8000/v1",
        whether_money=False,
):
    """关键函数:调用 OpenAI GPT-3 API 获取响应
      ① 判断模型类型（instruct 模型 vs 聊天模型）
      ② 创建 ChatAgent（根据模型类型配置）
      ③ 调用 agent.step() 获取响应
      ④ 尝试解析结构化输出（BDI：Belief, Desire, Intention）
      ⑤ 如果解析失败，调用 critic agent 提取关键信息
      ⑥ 返回：(结果，完整内容，结构化字典，输入内容)
  """
    content = ""
    input_content = {}
    if model_type in ["GPT_4"
        # ExtendedModelType.INSTRUCT_GPT,
        # ExtendedModelType.GPT_3_5_TURBO_INSTRUCT,
    ]:
        message = role.content + first_message.content + extra_prompt
        final_res = str_mes(gpt3_res(message, model_type.value))
        info = {}
    else:
        role = str_mes(role.content + extra_prompt)
        model_config = ChatGPTConfig(temperature=TEMPERATURE)

        agent = ChatAgent(
            role,
            model_type=model_type,
            output_language="English",
            model_config=model_config,
        )
        final_all_res = agent.step(first_message)
        final_res = final_all_res.msg
        info = final_all_res.info
        input_content["role"] = role.content
        input_content["input_message"] = first_message.content
    content += final_res.content
    if "fc" in info:
        structured_dict = json.loads(final_res.content)
        res = list(structured_dict.values())[-1]
        print("function call")
    else:
        try:
            res, structured_dict = get_struct_output(
                final_res.content, whether_money, test=True
            )
        except json.decoder.JSONDecodeError:
            res = cri_agent.step(final_res).msg.content
            structured_dict = {}
        except pydantic_core._pydantic_core.ValidationError:
            res = cri_agent.step(final_res).msg.content
            structured_dict = {}
    print(content)

    return (res, content, structured_dict, input_content)


def gen_character_res(
        all_chara_prompt,
        prompt_list,
        description,
        model_type,
        extra_prompt,
        whether_money,
        special_prompt,
):
    """
    遍历所有角色 → 创建 critic agent →
  组合角色提示词 → 调用 get_res() →
  收集结果和对话历史 → 处理 API 错误（超时/重试）
  """
    res = []
    dialog_history = []
    num = 0
    all_chara_prompt = list(all_chara_prompt)
    structured_output = []
    cha_num = 0
    while cha_num < len(all_chara_prompt):
        role = all_chara_prompt[cha_num]
        cri_agent = ChatAgent(
            BaseMessage(
                role_name="critic",
                role_type=RoleType.USER,
                meta_dict={},
                content=prompt_list[1],
            ),
            model_type=ExtendedModelType.QWEN3_5_FLASH,  # TODO Change if you need
            output_language="English",
        )
        role = role + like_people + special_prompt

        role_message = BaseMessage(
            role_name="player",
            role_type=RoleType.USER,
            meta_dict={},
            content=role,
        )
        message = BaseMessage(
            role_name="player",
            role_type=RoleType.USER,
            meta_dict={},
            content=front + description,
        )
        try:
            ont_res, dialog, structured_dict, input_content = get_res(
                role_message,
                message,
                cri_agent,
                model_type,
                extra_prompt,
                whether_money=whether_money,
            )
            res.append(ont_res)
            dialog_history.append([num, role, dialog])
            structured_output.append([structured_dict, input_content])
            num += 1
        except openai.APIError:
            time.sleep(30)
            cha_num -= 1
            print("API error")
        except openai.Timeout:
            time.sleep(30)
            print("Time out error")
            cha_num -= 1
        cha_num += 1
        print(cha_num)

    return res, dialog_history, structured_output


def save_json(prompt_list, data, model_type, k, save_path):
    if "lottery_problem" in prompt_list[0]:
        with open(
                save_path
                + prompt_list[0]
                + "_"
                + str(k)[:-1]
                + "_"
                + str(model_type.value)
                + "_lottery"
                + str(k)
                + ".json",
                "w",
        ) as json_file:
            json.dump(data, json_file)
    else:
        with open(
                save_path + prompt_list[0] + "_" +
                str(model_type.value) + ".json",
                "w",
        ) as json_file:
            json.dump(data, json_file)
    print(f"save {prompt_list[0]}")


def MAP(
        all_chara_prompt,
        prompt_list,
        model_type=ExtendedModelType.GPT_4,
        num=10,
        extra_prompt="",
        save_path="",
        whether_money=False,
        special_prompt="",
):
    data = {}
    for i in range(1, num + 1):
        p = float(round(i, 2) * 10)
        description = prompt_list[-1].format(p=f"{p}%", last=f"{100 - p}%")
        res, dialog_history, structured_output = gen_character_res(
            all_chara_prompt,
            prompt_list,
            description,
            model_type,
            extra_prompt,
            whether_money,
            special_prompt,
        )
        rate = sum([item == "trust" for item in res]) / len(res)
        res = {
            "p": p,
            "rate": rate,
            "res": res,
            "dialog": dialog_history,
            "origin_prompt": prompt_list,
            "structured_output": structured_output,
        }
        data[f"{p}_time_{i}"] = res
    with open(
            save_path + prompt_list[0] + "_" + str(model_type.value) + ".json",
            "w",
    ) as json_file:
        json.dump(data, json_file)


def agent_trust_experiment(
        all_chara_prompt,
        prompt_list,
        model_type=ExtendedModelType.GPT_4,
        k=3,
        extra_prompt="",
        save_path="",
        whether_money=False,
        special_prompt="",
):
    """
    功能：执行单次信任博弈实验
      ① 准备实验描述
      ② 调用 gen_character_res() 获取所有角色响应
      ③ 打包结果（包含原始数据、对话、结构化输出）
      ④ 保存为 JSON 文件
    """
    if "lottery_problem" in prompt_list[0]:
        description = prompt_list[-1].format(k=k)
    else:
        description = prompt_list[-1]
    res, dialog_history, structured_output = gen_character_res(
        all_chara_prompt,
        prompt_list,
        description,
        model_type,
        extra_prompt,
        whether_money,
        special_prompt,
    )
    data = {
        "res": res,
        "dialog": dialog_history,
        "origin_prompt": prompt_list,
        "structured_output": structured_output,
    }
    save_json(prompt_list, data, model_type, k, save_path)


def gen_intial_setting(
        model,
        ori_folder_path,
        LLM_Player=False,
        gender=None,
        extra_prompt="",
        prefix="",
        multi=False,
):
    """
    功能：生成实验初始设置
        处理逻辑：
        根据 LLM_Player 切换提示词
        根据 gender 修改角色描述
        添加 BDI 输出要求
        创建结果文件夹
    """
    global all_prompt
    all_prompt = copy.deepcopy(all_prompt_copy)
    folder_path = ori_folder_path
    if LLM_Player:
        all_prompt = llm_player_prompt
        folder_path = "LLM_player_" + ori_folder_path

    if gender is not None:
        for key, value in all_prompt.items():
            all_prompt[key][2] = value[2].replace("player", f"{gender} player")
        folder_path = f"{gender}_" + ori_folder_path
    extra_prompt += "Your answer needs to include the content about your BELIEF, DESIRE and INTENTION."

    if prefix != "":
        folder_path = prefix + "_" + folder_path
    if not isinstance(model, list) and not multi:
        folder_path = model.value + "_res/" + folder_path
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"folder {folder_path} is created")
        except OSError as e:
            print(f"creating folder {folder_path} failed:{e}")
    else:
        print(f"folder {folder_path} exists")

    return folder_path, extra_prompt


def run_exp(
        model_list,  # 模型列表，用于指定要运行实验的模型
        whether_llm_player=False,  # 是否使用LLM作为玩家的标志，默认为False
        gender=None,  # 性别参数，用于实验设置，默认为None
        special_prompt_key="",  # 特殊提示词的键，用于获取特定提示，默认为空字符串
        re_run=False,  # 是否重新运行已存在的实验，默认为False
        part_exp=True,  # 是否只运行部分实验，默认为True
        need_run=None,  # 指定需要运行的实验，默认为None
):
    """ 运行实验的主要函数
  遍历模型列表 →
    获取特殊提示词 →
    生成文件夹路径 →
    检查已存在结果 →
    遍历所有实验类型 →
      根据实验类型添加特定提示词（金钱/选择） →
      跳过已存在的实验 →
      调用对应实验函数：
        - MAP(): 实验 4,5,6（概率相关）
        - agent_trust_experiment(): 实验 7,9（特定概率）
        - 默认：其他实验

     """
    # 遍历模型列表中的每个模型
    for model in model_list:
        # 如果有特殊提示词键，则获取对应的特殊提示词
        if special_prompt_key != "":
            special_prompt = feature_prompt[special_prompt_key]
        else:
            special_prompt = ""  # 否则使用空字符串作为特殊提示词
        # 生成实验结果的文件夹路径
        folder_path = f"res/{model.value}_res/"
        # 生成初始设置，包括LLM玩家标志、性别和前缀提示词
        folder_path, extra_prompt = gen_intial_setting(
            model,
            folder_path,
            LLM_Player=whether_llm_player,
            gender=gender,
            prefix=special_prompt_key,
        )

        # 获取文件夹中已存在的JSON结果文件
        existed_res = [item for item in os.listdir(
            folder_path) if ".json" in item]
        # 遍历所有提示词
        for k, v in all_prompt.items():
            whether_money = False  # 标记是否需要输出金钱数量
            # 如果是部分实验模式且不需要运行特定实验，则跳过某些实验
            if k not in ["1", "2"] and part_exp and need_run is None:
                continue
            # 如果指定了需要运行的实验，则只运行指定的实验
            if need_run is not None:
                if k not in need_run:
                    continue
            # 对于特定实验，添加额外的提示要求输出金钱数量
            if k in ["1", "2", "8"]:
                extra_prompt = (
                        extra_prompt
                        + "You must end with 'Finally, I will give ___ dollars ' (numbers are required in the spaces)."
                )
                whether_money = True
            # 对于其他实验，添加额外的提示要求选择"Trust"或"not Trust"
            elif k in ["3", "4", "5", "6", "7", "9"]:
                extra_prompt = (
                        extra_prompt
                        + "You must end with 'Finally, I will choose ___' ('Trust' or 'not Trust' are required in the spaces)."
                )
            # 如果结果文件已存在且不需要重新运行，则跳过
            if check_file_if_exist(existed_res, v[0]) and not re_run:
                print(f"{v[0]} has existed")
                continue
            # 打印额外提示信息
            print("extra_prompt", extra_prompt)
            # 根据实验类型调用不同的实验函数
            if k in ["4", "5", "6"]:
                # 对于特定实验，使用MAP函数
                MAP(
                    all_chara_prompt,
                    v,
                    model,
                    extra_prompt=extra_prompt,
                    save_path=folder_path,
                    whether_money=whether_money,
                    special_prompt=special_prompt,
                )
            elif k in ["7", "9"]:
                # 对于其他特定实验，遍历不同概率值调用agent_trust_experiment
                for pro in ["46%"]:
                    agent_trust_experiment(
                        all_chara_prompt,
                        v,
                        model,
                        pro,
                        extra_prompt=extra_prompt,
                        save_path=folder_path,
                        whether_money=whether_money,
                        special_prompt=special_prompt,
                    )
            else:
                # 对于默认实验，直接调用agent_trust_experiment
                agent_trust_experiment(
                    all_chara_prompt,
                    v,
                    model,
                    extra_prompt=extra_prompt,
                    save_path=folder_path,
                    whether_money=whether_money,
                    special_prompt=special_prompt,
                )


def multi_round_exp(
        model_list,  # 模型列表，可以是单个模型或模型列表
        exp_time=1,  # 实验重复次数，默认为1
        round_num_inform=True,  # 是否在实验中显示轮次信息，默认为True
):
    """
    执行多轮实验的主函数，针对给定的模型列表进行多轮测试
    参数:
        model_list: 要测试的模型或模型列表
        exp_time: 每个模型要重复实验的次数
        round_num_inform: 是否在输出中显示轮次信息
    """
    for model in model_list:
        # 初始化前缀字符串
        prefix = ""
        # 如果模型是列表形式，遍历所有模型并组合它们的值作为前缀
        if isinstance(model, list):
            for i in model:
                prefix += prefix + i.value + "_"  # 拼接模型值到前缀
        else:
            prefix = model.value  # 如果是单个模型，直接使用其值作为前缀
        # 根据是否显示轮次信息，创建不同的结果文件夹路径
        folder_path = f"multi_res/{prefix}_res/"

        if not round_num_inform:
            folder_path = f"multi_no_round_num_res/{prefix}_res/"
        # 生成初始设置，获取文件夹路径和额外的提示信息
        folder_path, extra_prompt = gen_intial_setting(
            model,
            folder_path,
            multi=True,
        )
        # 使用t进度条进行多轮实验
        for i in tqdm.trange(exp_time):
            # 执行多轮实验，传入模型、角色列表、文件夹路径等参数
            multi_round(
                model,
                list(multi_round_chara_prompt),
                folder_path,
                prompt=round_prompt,
                round_num=5,
                exp_num=i + 1,
                round_num_inform=round_num_inform,
            )


if __name__ == "__main__":
    """
    关键特性:
        BDI 理论框架：要求模型输出 Belief（信念）、Desire（欲望）、Intention（意图）
        错误处理机制：API 超时/错误时自动重试
        结果去重：检查已存在的实验避免重复运行
        灵活配置：支持性别、种族、特征等多种变量
        结构化输出：使用 instructor 库解析标准化数据

    """
    model_list = [
        ExtendedModelType.QWEN3_5_FLASH,
        ExtendedModelType.GPT_4,
    ]

    # 非重复信任博弈
    # run_exp(model_list, part_exp=False)
    # llm experiment 测试 LLM 作为玩家
    # run_exp(model_list, whether_llm_player=1)
    # Gender 性别实验：male/female 角色
    # run_exp(model_list, gender="male")
    # run_exp(model_list, gender="female")
    # # Race 种族实验：5 种族裔对比
    # for race in race_list:
    #     run_exp(model_list, gender=race)
    # # Feature res 特征实验：不同人格特征
    # for k, v in feature_prompt.items():
    #     run_exp(model_list, special_prompt_key=k)

    # Muli experiment 多轮重复信任博弈实验

    # exp_time = 1
    #
    # model_list = [
    #     ExtendedModelType.GPT_3_5_TURBO_16K_0613,
    #     ExtendedModelType.GPT_4,
    # ]
    multi_round_exp(
        model_list, exp_time=1, round_num_inform=True
    )
