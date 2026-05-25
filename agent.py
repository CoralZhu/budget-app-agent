import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from db import get_all_categories, get_category_average, get_recent_transactions


SYSTEM_PROMPT = """你是一个消费异常检测助手。当用户询问异常消费时，你必须按以下步骤工作：
1. 先调用 get_recent_transactions 查最近 7 天的交易
2. 对于每个出现的分类，调用 get_category_average 查该分类的历史日均
3. 判断标准：单笔消费金额 > 该分类日均的 2 倍，视为异常
4. 用中文输出异常清单，说明每笔异常的金额、商家、超出基线多少
5. 如果没有异常，明确告诉用户'最近 7 天消费正常'
不要凭空猜测，所有判断必须基于工具返回的真实数据。"""


# OpenAI function calling 的 tools schema。
# DeepSeek 兼容 OpenAI Chat Completions 接口，所以这里沿用 OpenAI 的 function 格式。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_recent_transactions",
            "description": (
                "查询某个用户最近 N 天的支出交易记录。当用户要求检查最近消费、"
                "异常消费、近 7 天消费情况时，必须先调用这个工具获取真实交易数据。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "用户 ID。本示例主程序中固定使用 user_id=1。",
                    },
                    "days": {
                        "type": "integer",
                        "description": "查询最近多少天的交易；异常检测任务固定传 7。",
                    },
                },
                "required": ["user_id", "days"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_category_average",
            "description": (
                "查询某个用户某个消费分类的历史日均消费，用作异常检测基线。"
                "拿到最近 7 天交易后，应对每个出现过的分类调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "用户 ID。本示例主程序中固定使用 user_id=1。",
                    },
                    "category": {
                        "type": "string",
                        "description": "消费分类名称，例如：餐饮、购物、交通、娱乐。",
                    },
                    "months": {
                        "type": "integer",
                        "description": "统计最近多少个月的历史数据；默认使用 3 个月。",
                    },
                },
                "required": ["user_id", "category", "months"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_categories",
            "description": (
                "查询某个用户所有出现过的消费分类。仅当需要了解用户有哪些历史分类、"
                "但当前交易数据不足以判断分类范围时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "用户 ID。本示例主程序中固定使用 user_id=1。",
                    }
                },
                "required": ["user_id"],
                "additionalProperties": False,
            },
        },
    },
]


TOOL_FUNCTIONS = {
    "get_recent_transactions": get_recent_transactions,
    "get_category_average": get_category_average,
    "get_all_categories": get_all_categories,
}


def build_client() -> OpenAI | None:
    """读取 .env 中的 DeepSeek API Key，并创建兼容 OpenAI SDK 的客户端。"""
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：未在 .env 中找到 DEEPSEEK_API_KEY，无法调用 DeepSeek API。")
        return None

    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def to_json_text(data: Any) -> str:
    """把工具返回值转成 JSON 字符串，确保中文不被转义，方便 LLM 继续读取。"""
    return json.dumps(data, ensure_ascii=False, default=str)


def execute_tool(tool_name: str, arguments_text: str) -> str:
    """
    执行 LLM 请求的 tool。

    ReAct 中的 Action/Observation 对应到 OpenAI function calling：
    - Action 是 tool_name + arguments_text
    - Observation 是函数执行后的 JSON 结果
    """
    print(f"Tool Call: {tool_name}")
    print(f"Tool Args: {arguments_text}")

    if tool_name not in TOOL_FUNCTIONS:
        error_text = f"错误：未知工具 {tool_name}"
        print(f"Tool Result: {error_text}")
        return to_json_text({"error": error_text})

    try:
        arguments = json.loads(arguments_text or "{}")
    except json.JSONDecodeError as exc:
        error_text = f"错误：JSON 参数解析失败：{exc}"
        print(f"Tool Result: {error_text}")
        return to_json_text({"error": error_text, "raw_arguments": arguments_text})

    try:
        print(f"开始执行工具 {tool_name} ...")
        result = TOOL_FUNCTIONS[tool_name](**arguments)
    except Exception as exc:
        error_text = f"错误：工具 {tool_name} 执行失败：{exc}"
        print(f"Tool Result: {error_text}")
        return to_json_text({"error": error_text})

    result_text = to_json_text(result)
    print(f"Tool Result: {result_text}")
    return result_text


def run_agent(user_id: int, user_message: str, max_rounds: int = 10) -> None:
    """
    手写 ReAct 循环。

    每一轮都把历史 messages、tools 传给模型：
    - 如果模型返回 tool_calls，说明它还需要查数据；执行工具后把结果追加为 tool message
    - 如果模型返回普通文本，说明它已经完成分析；打印最终答案并退出
    - 最多跑 max_rounds 轮，避免模型反复调用工具导致死循环
    """
    client = build_client()
    if client is None:
        return

    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n当前用户 ID 是 {user_id}。"},
        {"role": "user", "content": user_message},
    ]

    round_index = 1
    while True:
        if round_index > max_rounds:
            print(f"\n==== 已达到最大轮数 {max_rounds}，为防止死循环已停止 ====")
            return

        print(f"\n==== Round {round_index} ====")
        print("Thought: 正在请求 DeepSeek，让模型决定是调用工具还是输出最终答案。")

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS,
            )
        except OpenAIError as exc:
            print(f"错误：DeepSeek API 调用失败：{exc}")
            return
        except Exception as exc:
            print(f"错误：调用 DeepSeek 时发生未知异常：{exc}")
            return

        assistant_message = response.choices[0].message
        assistant_content = assistant_message.content or ""
        tool_calls = assistant_message.tool_calls or []

        print(f"Thought: {assistant_content or '模型未返回文本思考，直接给出了工具调用。'}")

        if not tool_calls:
            print("\n==== Final Answer ====")
            print(assistant_content)
            return

        # 把 assistant 的 tool_calls 也加入 messages，后续 tool message 才能和对应 call_id 对齐。
        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
        )

        for tool_call in tool_calls:
            result_text = execute_tool(
                tool_call.function.name,
                tool_call.function.arguments,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": result_text,
                }
            )

        round_index += 1


if __name__ == "__main__":
    USER_ID = 1
    run_agent(USER_ID, "帮我检查最近 7 天有没有异常消费")
