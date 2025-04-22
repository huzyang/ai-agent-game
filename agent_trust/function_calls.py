import json

from camel.functions.openai_function import OpenAIFunction


def trust_or_not_FC(Believe, Desire, Intention, Trust_or_not, Risk, Strategy, Think):
    """
    根据信念、欲望、意图、信任选择、风险评估、策略和思考过程，生成信任决策的结果。
    
    参数:
        Believe (any): 信念因素。
        Desire (any): 欲望因素。
        Intention (any): 意图因素。
        Trust_or_not (any): 是否信任的选择。
        Risk (any): 风险评估。
        Strategy (any): 考虑的策略。
        Think (any): 思考过程或推理。
    
    返回:
        Dict[str, Any]: 包含模型答案的字典，键包括 Believe, Desire, Intention, Trust_or_not, Risk, Strategy, Think。
    """
    model_answer = {
        "Believe": Believe,
        "Desire": Desire,
        "Intention": Intention,
        "Trust_or_not": Trust_or_not,
        "Risk": Risk,
        "Strategy": Strategy,
        "Think": Think
    }
    return model_answer


def given_money_FC(Believe, Desire, Intention, money_num, Risk, Strategy, Think):
    """
    根据信念、欲望、意图、金额数量、风险评估、策略和思考过程，生成金钱分配的结果。
    
    参数:
        Believe (any): 信念因素。
        Desire (any): 欲望因素。
        Intention (any): 意图因素。
        money_num (any): 考虑的金额数量。
        Risk (any): 与金额相关的风险评估。
        Strategy (any): 与金额相关的策略。
        Think (any): 关于金额决策的思考过程或推理。
    
    返回:
        Dict[str, Any]: 包含模型答案的字典，键包括 Believe, Desire, Intention, money_num, Risk, Strategy, Think。
    """
    model_answer = {
        "Believe": Believe,
        "Desire": Desire,
        "Intention": Intention,
        "money_num": money_num,
        "Risk": Risk,
        "Strategy": Strategy,
        "Think": Think
    }
    return model_answer


money_paramters = {
    # 定义用于描述金钱相关参数的 JSON Schema。
    "type": "object",
    "properties": {
        "Believe": {
            "type": "string",
            "description": "What's your Believe?",
        },
        "Desire": {
            "type": "string",
            "description": "What do you desire?",
        },
        "Intention": {
            "type": "string",
            "description": "What's your Intention?",
        },
        "money_num": {
            "type": "string",
            "description": "How much money would you give each other",
        },
        "Risk": {
            "type": "string",
            "description": "What is the potential risk in the game?"
        },
        "Strategy": {
            "type": "string",
            "description": " what is the potential strategies in the game?"
        },
        "Think": {
            "type": "string",
            "description": "The thinking progress in this game"
        }
    },
    "required": ["Believe", "Desire", "Intention", "money_num", "Risk", "Strategy", "Think"],
}

trust_paramters = {
    # 定义用于描述信任相关参数的 JSON Schema。
    "type": "object",
    "properties": {
        "Believe": {
            "type": "string",
            "description": "What's your Believe?",
        },
        "Desire": {
            "type": "string",
            "description": "What do you desire?",
        },
        "Intention": {
            "type": "string",
            "description": "What's your Intention?",
        },
        "Trust_or_not": {
            "type": "string",
            "description": "Do you trust each other? Only responce 'trust' or 'not trust'",
        },
        "Risk": {
            "type": "string",
            "description": "What is the potential risk in the game?"
        },
        "Strategy": {
            "type": "string",
            "description": " what is the potential strategies in the game?"
        },
        "Think": {
            "type": "string",
            "description": "The thinking progress in this game"
        }
    },
    "required": ["Believe", "Desire", "Intention", "Trust_or_not", 'Risk', 'Strategy', "Think"],
}


def get_function_call_res(message):
    """
    解析消息中的函数调用信息，并执行对应的函数。
    
    参数:
        message (Dict): 包含函数调用信息的消息字典。
    
    返回:
        Any: 执行函数后的结果。
    """
    if message.get("function_call"):
        function_name = message["function_call"]["name"]
        ans = json.loads(message["function_call"]["arguments"])
        func = globals().get(function_name)
        res = func(**ans)

        return res


money_call = OpenAIFunction(
    # 定义与金钱分配相关的 OpenAIFunction 实例。
    func=given_money_FC,
    name="given_money_FC",
    description="This function is need when inquiring about the amount of money to give.",
    parameters=money_paramters,
)

trust_call = OpenAIFunction(
    # 定义与信任决策相关的 OpenAIFunction 实例。
    func=trust_or_not_FC,
    name="trust_or_not_FC",
    description="You choose to trust each other or not trust each other?",
    parameters=trust_paramters,
)

function_list = [money_call.as_dict(), trust_call.as_dict()]
