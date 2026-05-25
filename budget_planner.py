import json
from datetime import datetime
from typing import Any

from openai import OpenAIError

from agent import build_client, to_json_text
from db import (
    get_current_budgets,
    get_historical_average_by_category,
    get_monthly_income,
    get_monthly_spending_by_category,
    save_budget_plan,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_historical_average_by_category",
            "description": (
                "查询用户过去 N 个月每个消费分类的月均消费。"
                "当需要基于真实历史消费生成预算建议、判断各分类合理额度时调用。"
                "参数 user_id 为用户 ID，months 为回看月份数，通常传 3。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "用户 ID。"},
                    "months": {
                        "type": "integer",
                        "description": "回看最近多少个月，预算规划默认使用 3。",
                    },
                },
                "required": ["user_id", "months"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_monthly_income",
            "description": (
                "查询用户某个月的总收入。"
                "当需要判断预算是否超过收入、是否需要动用积蓄、或制定月度预算上限时调用。"
                "year_month 必须是 YYYY-MM 格式，例如 2026-05。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "用户 ID。"},
                    "year_month": {
                        "type": "string",
                        "description": "目标月份，格式为 YYYY-MM。",
                    },
                },
                "required": ["user_id", "year_month"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_budgets",
            "description": (
                "查询用户某个月已经保存的总预算和分类预算。"
                "当用户要为某月做预算时，应检查是否已有预算，避免覆盖用户已有方案。"
                "year_month 必须是 YYYY-MM 格式。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "用户 ID。"},
                    "year_month": {
                        "type": "string",
                        "description": "目标月份，格式为 YYYY-MM。",
                    },
                },
                "required": ["user_id", "year_month"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_monthly_spending_by_category",
            "description": (
                "查询用户某个月每个分类已经花了多少钱。"
                "当预算月份是当前月、或需要参考本月已发生支出以调整剩余额度时调用。"
                "year_month 必须是 YYYY-MM 格式。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "用户 ID。"},
                    "year_month": {
                        "type": "string",
                        "description": "目标月份，格式为 YYYY-MM。",
                    },
                },
                "required": ["user_id", "year_month"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_budget_plan",
            "description": (
                "写入或更新用户某个月的预算方案。参数 total_budget 是总预算，"
                "category_budgets 是分类预算列表，每项包含 category 和 amount。"
                "重要：仅在用户明确确认（说了\"确认\"、\"保存\"、\"写入\"、\"就这样\"等明确表态后）才能调用此工具。"
                "不能在用户还在调整方案时调用。不能未经用户确认擅自写入。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "用户 ID。"},
                    "year_month": {
                        "type": "string",
                        "description": "目标月份，格式为 YYYY-MM。",
                    },
                    "total_budget": {
                        "type": "number",
                        "description": "该月总预算金额。",
                    },
                    "category_budgets": {
                        "type": "array",
                        "description": "分类预算列表。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "description": "消费分类名称。",
                                },
                                "amount": {
                                    "type": "number",
                                    "description": "该分类预算金额。",
                                },
                            },
                            "required": ["category", "amount"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["user_id", "year_month", "total_budget", "category_budgets"],
                "additionalProperties": False,
            },
        },
    },
]


TOOL_FUNCTIONS = {
    "get_historical_average_by_category": get_historical_average_by_category,
    "get_monthly_income": get_monthly_income,
    "get_current_budgets": get_current_budgets,
    "get_monthly_spending_by_category": get_monthly_spending_by_category,
    "save_budget_plan": save_budget_plan,
}


CONFIRM_WORDS = ("确认", "保存", "写入", "就这样", "可以", "同意", "没问题", "按这个")
NEGATIVE_WORDS = ("不要", "别", "先不", "暂不", "取消", "不保存", "别写入")


def build_system_prompt(user_id: int) -> str:
    """生成预算规划 Agent 的 system prompt，把用户 ID 和当前日期写入上下文。"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    return f"""你是一个个性化预算规划助手。

用户 ID: {user_id}
当前日期: {current_date}

你必须按以下流程工作：
Step 1: 用户启动对话后，你主动开口问："你想为哪个月做预算？有储蓄目标吗？"
Step 2: 用户说明月份和目标后，调用 get_historical_average_by_category 看过去 3 个月历史月均。
Step 3: 调用 get_monthly_income 看该月份收入水平。
Step 4: 调用 get_current_budgets 看该月份现有预算（如果有）。
Step 5: 必要时调用 get_monthly_spending_by_category 看该月已经花掉的金额，尤其是当前月预算。
Step 6: 综合真实数据，生成预算方案，并用自然语言展示给用户，例如：
基于你过去3个月数据，建议下个月预算：
- 餐饮 ¥800（月均 ¥396 × 2 倍缓冲）
- 购物 ¥700（月均 ¥630 + 略压缩）
总预算 ¥3500，按你 ¥2000 收入算需要从积蓄补 ¥1500
Step 7: 询问用户：是否接受？想调整哪些分类？
Step 8: 用户调整后，再次展示最终方案。
Step 9: 只有当用户明确说“确认/写入/保存/就这样”等关键词后，才能调用 save_budget_plan 写入数据库。
Step 10: 写入成功后，简短确认并结束对话。

重要约束：
- 必须基于工具返回的真实历史数据给建议，不能凭空编造金额。
- 写入数据库前必须得到用户明确确认。
- 用户还在讨论、比较、调整方案时，绝对不能调用 save_budget_plan。
- 所有回复使用中文，语气友好、清楚、简洁。
- 如果信息不足，先追问；不要猜测目标月份、储蓄目标或用户意图。"""


def summarize_tool_result(result: Any) -> str:
    """生成简短调试摘要，避免把很长的工具结果全部刷到屏幕上。"""
    if isinstance(result, list):
        preview = result[:2]
        return f"返回 {len(result)} 条，预览: {to_json_text(preview)}"
    if isinstance(result, dict):
        return f"返回字段 {list(result.keys())}: {to_json_text(result)[:240]}"
    return f"返回: {result}"


def last_user_message(messages: list[dict[str, Any]]) -> str:
    """取最近一条用户消息，用于 save_budget_plan 的代码侧确认检查。"""
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def has_explicit_confirmation(messages: list[dict[str, Any]]) -> bool:
    """判断最近用户输入是否包含明确保存确认，并排除明显否定表达。"""
    text = last_user_message(messages)
    if any(word in text for word in NEGATIVE_WORDS):
        return False
    return any(word in text for word in CONFIRM_WORDS)


def execute_budget_tool(tool_name: str, arguments_text: str, messages: list[dict[str, Any]]) -> str:
    """执行预算规划 Agent 的工具调用，并把结果转成 tool message 需要的 JSON 文本。"""
    print(f"🔧 Agent 正在调用: {tool_name}")

    if tool_name not in TOOL_FUNCTIONS:
        error_text = f"错误：未知工具 {tool_name}"
        print(f"工具结果摘要：{error_text}")
        return to_json_text({"error": error_text})

    if tool_name == "save_budget_plan" and not has_explicit_confirmation(messages):
        error_text = "错误：用户尚未明确确认，禁止调用 save_budget_plan 写入数据库。"
        print(f"工具结果摘要：{error_text}")
        return to_json_text({"error": error_text})

    try:
        arguments = json.loads(arguments_text or "{}")
    except json.JSONDecodeError as exc:
        error_text = f"错误：JSON 参数解析失败：{exc}"
        print(f"工具结果摘要：{error_text}")
        return to_json_text({"error": error_text, "raw_arguments": arguments_text})

    try:
        result = TOOL_FUNCTIONS[tool_name](**arguments)
    except Exception as exc:
        error_text = f"错误：工具 {tool_name} 执行失败：{exc}"
        print(f"工具结果摘要：{error_text}")
        return to_json_text({"error": error_text})

    print(f"工具结果摘要：{summarize_tool_result(result)}")
    return to_json_text(result)


def append_assistant_with_tool_calls(messages: list[dict[str, Any]], assistant_message: Any) -> None:
    """把 assistant 的 tool_calls 原样记入历史，后续 tool message 才能按 tool_call_id 对齐。"""
    messages.append(
        {
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in assistant_message.tool_calls or []
            ],
        }
    )


def model_until_text(client: Any, messages: list[dict[str, Any]], max_tool_rounds: int = 10) -> bool:
    """
    内层模型推理循环。

    模型可能连续调用多个工具，因此这里会一直请求模型：
    - 有 tool_calls：执行工具，把结果追加为 tool message，然后继续请求模型
    - 没有 tool_calls：打印 assistant 文本，加入历史，并跳出内层循环
    """
    tool_round = 1
    while True:
        if tool_round > max_tool_rounds:
            print("已达到单次对话最大工具轮数，为防止死循环，本轮已停止。")
            return False

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS,
            )
        except OpenAIError as exc:
            print(f"错误：DeepSeek API 调用失败：{exc}")
            return False
        except Exception as exc:
            print(f"错误：调用 DeepSeek 时发生未知异常：{exc}")
            return False

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            assistant_text = assistant_message.content or ""
            messages.append({"role": "assistant", "content": assistant_text})
            print(f"\n预算助手: {assistant_text}")
            return True

        append_assistant_with_tool_calls(messages, assistant_message)
        for tool_call in tool_calls:
            result_text = execute_budget_tool(
                tool_call.function.name,
                tool_call.function.arguments,
                messages,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": result_text,
                }
            )

        tool_round += 1


def run_conversation(user_id: int) -> None:
    """
    多轮对话主循环。

    外层循环负责读取用户输入并维护长期 messages 记忆；
    内层循环负责让模型完成当前轮推理，必要时多次调用工具，直到给出文本回复。
    """
    client = build_client()
    if client is None:
        return

    messages = [{"role": "system", "content": build_system_prompt(user_id)}]

    # 会话启动时先让 Agent 主动开口，完成 Step 1。
    if not model_until_text(client, messages):
        return

    while True:
        user_text = input("\n你: ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("预算助手: 好的，预算规划对话已结束。")
            return

        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        if not model_until_text(client, messages):
            return


if __name__ == "__main__":
    USER_ID = 1
    print("欢迎使用多轮预算规划 Agent。输入 exit 或 quit 可以退出。")
    run_conversation(USER_ID)
