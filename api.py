import json
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAIError
from pydantic import BaseModel

import agent as anomaly_agent
import budget_planner
from agent import build_client, to_json_text
from budget_planner import (
    append_assistant_with_tool_calls,
    build_system_prompt,
    execute_budget_tool,
)


app = FastAPI(title="AI Budget Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnomalyCheckRequest(BaseModel):
    user_id: int
    days: int = 7


class AnomalyItem(BaseModel):
    transaction_id: int
    amount: float
    category: str
    merchant: str
    baseline: float
    ratio: float
    reason: str


class AnomalyCheckResponse(BaseModel):
    has_anomaly: bool
    anomalies: list[AnomalyItem]
    summary: str
    raw_response: str


class ChatMessage(BaseModel):
    role: str
    content: Any = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class BudgetChatRequest(BaseModel):
    user_id: int
    messages: list[ChatMessage]


class BudgetChatResponse(BaseModel):
    reply: str
    messages: list[dict[str, Any]]
    tool_calls_made: list[str]
    budget_saved: bool


class BudgetStartResponse(BaseModel):
    messages: list[dict[str, Any]]
    reply: str


def get_client_or_raise():
    """创建 DeepSeek/OpenAI 兼容客户端；API key 缺失时返回清晰的 500。"""
    client = build_client()
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="未配置 DEEPSEEK_API_KEY，请在 .env 中设置后重试。",
        )
    return client


def message_to_dict(message: ChatMessage) -> dict[str, Any]:
    """把 Pydantic message 转成 OpenAI SDK 可接受的 dict，并去掉 None 字段。"""
    if hasattr(message, "model_dump"):
        data = message.model_dump(exclude_none=True)
    else:
        data = message.dict(exclude_none=True)
    return data


def ensure_budget_system_message(
    user_id: int,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """无状态接口由前端传历史；如果缺 system message，这里自动补上。"""
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": build_system_prompt(user_id)}] + messages


def execute_anomaly_tool(tool_name: str, arguments_text: str) -> str:
    """执行异常检测 Agent 的工具调用，返回 tool message 所需 JSON 字符串。"""
    if tool_name not in anomaly_agent.TOOL_FUNCTIONS:
        return to_json_text({"error": f"未知工具 {tool_name}"})

    try:
        arguments = json.loads(arguments_text or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"工具参数 JSON 解析失败：{exc}")

    try:
        result = anomaly_agent.TOOL_FUNCTIONS[tool_name](**arguments)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"工具 {tool_name} 执行失败：{exc}")

    return to_json_text(result)


def append_tool_call_message(messages: list[dict[str, Any]], assistant_message: Any) -> None:
    """把带 tool_calls 的 assistant 消息写入上下文，保持 tool_call_id 对齐。"""
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


def run_anomaly_react(user_id: int, days: int, max_rounds: int = 10) -> str:
    """异常检测 HTTP 版 ReAct 循环：最终返回 assistant 的 JSON 文本。"""
    client = get_client_or_raise()
    json_prompt = """

完成分析后，最终输出必须严格按以下 JSON 格式：
{"has_anomaly": true/false, "anomalies": [{"transaction_id": 67, "amount": 124.6, "category": "购物", "merchant": "美团", "baseline": 21.02, "ratio": 5.9, "reason": "..."}], "summary": "..."}

关键要求：
- 每条异常必须带上 transaction_id 字段（即 get_recent_transactions 返回的 id 字段）
- 用户已经知道这笔异常的 transaction_id 后，前端会自动去重不再显示
- 不要任何额外文字，只输出 JSON
"""
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                f"{anomaly_agent.SYSTEM_PROMPT}\n\n"
                f"当前用户 ID 是 {user_id}。本次接口请求的检查窗口是最近 {days} 天；"
                f"调用 get_recent_transactions 时 days 必须传 {days}。"
                f"{json_prompt}"
            ),
        },
        {"role": "user", "content": f"帮我检查最近 {days} 天有没有异常消费"},
    ]

    for _ in range(max_rounds):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=anomaly_agent.TOOLS,
                response_format={"type": "json_object"},
            )
        except OpenAIError as exc:
            raise HTTPException(status_code=502, detail=f"DeepSeek API 调用失败：{exc}")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"调用 DeepSeek 时发生未知异常：{exc}")

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []
        if not tool_calls:
            return assistant_message.content or ""

        append_tool_call_message(messages, assistant_message)
        for tool_call in tool_calls:
            result_text = execute_anomaly_tool(
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

    raise HTTPException(status_code=500, detail="异常检测 Agent 达到最大循环次数，已停止。")


def parse_anomaly_response(raw_response: str) -> dict[str, Any]:
    """解析 LLM 最终 JSON；失败时把原始文本带回，方便排查 prompt 或模型输出问题。"""
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Agent 最终输出不是合法 JSON：{exc}",
                "raw_response": raw_response,
            },
        )

    data["raw_response"] = raw_response
    return data


def run_budget_until_text(
    client: Any,
    messages: list[dict[str, Any]],
    max_tool_rounds: int = 10,
) -> tuple[str, list[str]]:
    """
    预算规划 HTTP 版内层循环。

    复用 budget_planner 的 tools、tool 执行函数和 tool_calls 追加函数，
    但这里返回 reply 和本轮调用过的工具名，便于 API 响应给前端。
    """
    tool_calls_made: list[str] = []

    for _ in range(max_tool_rounds):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=budget_planner.TOOLS,
            )
        except OpenAIError as exc:
            raise HTTPException(status_code=502, detail=f"DeepSeek API 调用失败：{exc}")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"调用 DeepSeek 时发生未知异常：{exc}")

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            reply = assistant_message.content or ""
            messages.append({"role": "assistant", "content": reply})
            return reply, tool_calls_made

        append_assistant_with_tool_calls(messages, assistant_message)
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_calls_made.append(tool_name)
            result_text = execute_budget_tool(
                tool_name,
                tool_call.function.arguments,
                messages,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result_text,
                }
            )

    raise HTTPException(status_code=500, detail="预算规划 Agent 达到最大工具循环次数，已停止。")


@app.post("/api/agent/anomaly-check", response_model=AnomalyCheckResponse)
def anomaly_check(request: AnomalyCheckRequest) -> AnomalyCheckResponse:
    """异常消费检测：执行一轮 ReAct，解析最终 JSON 并返回前端友好结构。"""
    raw_response = run_anomaly_react(request.user_id, request.days)
    data = parse_anomaly_response(raw_response)
    return AnomalyCheckResponse(**data)


@app.post("/api/agent/budget/chat", response_model=BudgetChatResponse)
def budget_chat(request: BudgetChatRequest) -> BudgetChatResponse:
    """多轮预算规划：前端传完整历史，本接口补齐 system 后推进一轮。"""
    client = get_client_or_raise()
    messages = [message_to_dict(message) for message in request.messages]
    messages = ensure_budget_system_message(request.user_id, messages)

    reply, tool_calls_made = run_budget_until_text(client, messages)
    return BudgetChatResponse(
        reply=reply,
        messages=messages,
        tool_calls_made=tool_calls_made,
        budget_saved="save_budget_plan" in tool_calls_made,
    )


@app.get("/api/agent/budget/start", response_model=BudgetStartResponse)
def budget_start(user_id: int = Query(..., description="用户 ID")) -> BudgetStartResponse:
    """启动预算规划新对话：创建 system message，并让 Agent 主动开场。"""
    client = get_client_or_raise()
    messages = [{"role": "system", "content": build_system_prompt(user_id)}]
    reply, _ = run_budget_until_text(client, messages)
    return BudgetStartResponse(messages=messages, reply=reply)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
