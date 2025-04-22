import copy
import json
import os
import sys

import gradio as gr
import openai
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig, OpenSourceConfig
from camel.messages import BaseMessage
from camel.types import ModelType, RoleType
from exp_model_class import ExtendedModelType

# 定义开源模型路径字典，用于映射模型类型到具体的模型路径
open_model_path_dict = {
    ModelType.VICUNA: "lmsys/vicuna-7b-v1.3",
    ModelType.LLAMA_2: "meta-llama/Llama-2-7b-chat-hf",
}

# 前置提示语，用于强调角色是人而非AI模型
front = "you are a person not an ai model."

# 定义一个函数，将字符串内容包装为BaseMessage对象
def str_mes(content):
    return BaseMessage(
        role_name="player",
        role_type=RoleType.USER,
        meta_dict={},
        content=content,
    )

# 定义一个函数，调用OpenAI的Completion API生成文本
def gpt3_res(prompt, model_name="text-davinci-003", temperature=1):
    response = openai.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=1500,
    )
    return response.choices[0].text.strip()

# 核心函数，根据角色、游戏类型和其他参数生成响应内容
def get_res_for_visible(
    role,
    first_message,
    game_type,
    api_key,
    model_type=ExtendedModelType.GPT_4,
    extra_prompt="",
    temperature=1.0,
    player_demographic=None,
):
    # 初始化返回内容
    content = ""
    # 设置OpenAI API Key，优先使用传入的api_key，否则从环境变量中获取
    if api_key is not None or api_key != "":
        openai.api_key = api_key
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # 构建额外提示信息，包含BELIEF、DESIRE和INTENTION等内容
    extra_prompt += "Your answer needs to include the content about your BELIEF, DESIRE and INTENTION."
    if "game" in game_type.lower():
        extra_prompt += "You must end with 'Finally, I will give ___ dollars ' (numbers are required in the spaces)."
    else:
        extra_prompt += "You must end with 'Finally, I will choose ___' ('Trust' or 'not Trust' are required in the spaces)."
    extra_prompt += front

    # 将角色和额外提示信息封装为BaseMessage对象
    role = str_mes(role + extra_prompt)
    # 如果提供了玩家人口统计信息，则替换消息中的占位符
    if player_demographic is not None:
        first_message = first_message.replace(
            "player", player_demographic+" player")
    first_message = str_mes(first_message)

    # 根据模型类型选择不同的处理逻辑
    if model_type in [
        ExtendedModelType.INSTRUCT_GPT,
        ExtendedModelType.GPT_3_5_TURBO_INSTRUCT,
    ]:
        # 对于Instruct GPT模型，直接拼接消息并调用gpt3_res生成响应
        message = role.content + first_message.content + extra_prompt
        final_res = str_mes(gpt3_res(message, model_type.value, temperature))
    else:
        # 对于其他模型，使用ChatAgent进行处理
        role = str_mes(role.content + extra_prompt)
        model_config = ChatGPTConfig(temperature=temperature)
        if model_type in [
            ModelType.VICUNA,
            ModelType.LLAMA_2,
        ]:
            # 配置开源模型的参数
            open_source_config = dict(
                model_type=model_type,
                model_config=OpenSourceConfig(
                    model_path=open_model_path_dict[model_type],
                    server_url="http://localhost:8000/v1",
                    api_params=ChatGPTConfig(temperature=temperature),
                ),
            )
            agent = ChatAgent(
                role, output_language="English", **(open_source_config or {})
            )
        else:
            # 配置闭源模型的参数
            agent = ChatAgent(
                role,
                model_type=model_type,
                output_language="English",
                model_config=model_config,
            )
        # 调用ChatAgent的step方法生成响应
        final_all_res = agent.step(first_message)
        final_res = final_all_res.msg
    # 拼接最终响应内容
    content += final_res.content

    return content

# 添加上级目录到系统路径，以便导入其他模块
sys.path.append("../..")

# 定义角色信息和游戏提示文件路径
file_path_character_info = 'prompt/character_2.json'
file_path_game_prompts = 'prompt/person_all_game_prompt.json'

# 加载角色信息和游戏提示
with open(file_path_character_info, 'r') as file:
    character_info = json.load(file)

# Load game prompts
with open(file_path_game_prompts, 'r') as file:
    game_prompts = json.load(file)

# 提取角色名称和信息
characters = [f'Trustor Persona {i}' for i in range(
    1, len(character_info) + 1)]
character_info = {f'Trustor Persona {i}': info for i, info in enumerate(
    character_info.values(), start=1)}

# 提取游戏名称和提示
game_prompts = {
    prompt[0]: prompt[-1] for i, prompt in enumerate(game_prompts.values(), start=1)}
games = list(game_prompts.keys())
print(games)

# 定义模型映射字典，用于将模型名称映射到ExtendedModelType
model_dict = {
    'gpt-3.5-turbo-0613': ExtendedModelType.GPT_3_5_TURBO_0613,
    'gpt-3.5-turbo-16k-0613': ExtendedModelType.GPT_3_5_TURBO_16K_0613,
    'gpt-4': ExtendedModelType.GPT_4,
    'text-davinci-003': ExtendedModelType.INSTRUCT_GPT,
    'gpt-3.5-turbo-instruct': ExtendedModelType.GPT_3_5_TURBO_INSTRUCT,
    # 'vicuna': ModelType.VICUNA,
    # 'llama-2': ModelType.LLAMA_2,
}

# 定义游戏树图片路径字典
game_tree_images = {
    "Dictator_Game": "game_tree/dictator_game_game_tree.png",
    "Trust_Game": "game_tree/Trust_game_game_tree.png",
    "map_risky_dictator_problem": "game_tree/risky_dictator_game_game_tree.png",
    "map_trust_problem": "game_tree/map_trust_game_game_tree.png",
    "lottery_problem_people": "game_tree/lottery_people_game_tree.png",
    "lottery_problem_gamble": "game_tree/lottery_gamble_game_tree.png"
}

# 提取模型名称列表
models = list(model_dict.keys())

# 定义一个函数，根据角色名称更新角色信息显示
def update_char_info(char):
    return character_info.get(char, "No information available.")

# 定义一个函数，根据游戏名称更新游戏提示显示
def update_game_prompt(game):
    return game_prompts.get(game, "No prompt available.")

# 定义一个函数，处理用户提交的表单数据并生成结果
def process_submission(character, game, api_key=None,  model="gpt-3.5-turbo-0613",  extra_prompt="", temperature=1.0, player_demographic=None,):
    # 设置API Key，优先使用传入的api_key，否则从环境变量中获取
    if api_key is None or api_key == "":
        api_key = os.environ.get("OPENAI_API_KEY")
    else:
        os.environ["OPENAI_API_KEY"] = api_key
    # 调用get_res_for_visible生成响应内容
    return get_res_for_visible(character_info.get(character, ""), game_prompts.get(game, "No prompt available."), game, api_key, model_dict[model], extra_prompt, temperature, player_demographic)

# 定义一个函数，根据游戏名称更新游戏图片显示
def update_game_image(game_name):
    image_path = game_tree_images.get(game_name, None)
    return image_path

# 使用Gradio构建交互式界面
with gr.Blocks() as app:
    # 显示游戏介绍文本框
    game_introduction = gr.Textbox(
        label="Instruction", value="""1. You should select the persona for the trustor and the type of game.\n
2. You need to fill in your OpenAI API Key.\n
3. If you fill in 'Extra Prompt for Trustor', this prompt will be the additional system prompt to the trustor.\n
4. You can fill in the trustee player's demographics, such as race or gender.\n
5. If you want reset the conversation, please refresh this page.""")
    with gr.Row():
        # 角色下拉框，用于选择信任者的角色
        char_dropdown = gr.Dropdown(
            choices=characters, label="Select Trustor Persona", value=characters[0])
        # 游戏下拉框，用于选择游戏类型
        game_dropdown = gr.Dropdown(
            choices=games, label="Select Game")
    # 显示角色信息的文本框
    char_info_display = gr.Textbox(
        label="Trustor Persona Info", value=character_info[characters[0]])
    with gr.Row():
        # 显示游戏提示的文本框
        game_prompt_display = gr.Textbox(
            label="Game Prompt", value=game_prompts["Trust_Game"])
        # 显示游戏图片的图像框
        game_image_display = gr.Image(
            label="Game Image")

    # 输入OpenAI API Key的文本框
    api_key_input = gr.Textbox(
        label="OpenAI API Key", placeholder="Enter your OpenAI API Key here")
    # 模型选择下拉框
    model_dropdown = gr.Dropdown(
        choices=models, label="Select Model", value=models[0])
    # 额外提示输入框
    extra_prompt_input = gr.Textbox(
        label="Extra Prompt for Trustor", placeholder="Enter any additional prompt here (Optional)")
    # 温度调节滑块
    temperature_slider = gr.Slider(
        minimum=0.0, maximum=1.0, step=0.01, label="Temperature", value=1.0)
    # 玩家人口统计信息输入框
    player_demographic_input = gr.Textbox(
        label="Trustee Player Demographic", placeholder="Enter trustee player demographic info here (Optional)")
    # 提交按钮
    submit_button = gr.Button("Submit")
    # 显示结果的文本框
    result_display = gr.Textbox(label="Result")

    # 绑定事件：当角色下拉框值改变时，更新角色信息显示
    char_dropdown.change(
        update_char_info, inputs=char_dropdown, outputs=char_info_display)
    # 绑定事件：当游戏下拉框值改变时，更新游戏提示显示
    game_dropdown.change(update_game_prompt,
                         inputs=game_dropdown, outputs=game_prompt_display)
    # 绑定事件：当游戏下拉框值改变时，更新游戏图片显示
    game_dropdown.change(
        update_game_image, inputs=game_dropdown, outputs=game_image_display)

    # 绑定事件：当点击提交按钮时，处理表单数据并显示结果
    submit_button.click(
        process_submission,
        inputs=[char_dropdown, game_dropdown, api_key_input, model_dropdown,
                extra_prompt_input, temperature_slider, player_demographic_input],
        outputs=result_display
    )

# 启动Gradio应用
app.launch()