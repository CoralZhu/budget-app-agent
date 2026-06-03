import json
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAIError
from pydantic import BaseModel

import agent as anomaly_agent
import budget_planner
from agent import build_client, to_json_text
from auth import get_current_user_id
from budget_planner import (
    append_assistant_with_tool_calls,
    build_system_prompt,
    execute_budget_tool,
)
from db import (
    create_conversation,
    get_connection,
    get_conversation_messages,
    list_user_conversations,
    save_message,
    save_messages_batch,
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
    messages: list[ChatMessage]
    conversation_id: int | None = None


class BudgetChatResponse(BaseModel):
    reply: str
    messages: list[dict[str, Any]]
    tool_calls_made: list[str]
    budget_saved: bool
    conversation_id: int


class BudgetStartResponse(BaseModel):
    messages: list[dict[str, Any]]
    reply: str
    conversation_id: int


class ConversationSummary(BaseModel):
    id: int
    agent_type: str
    title: str | None
    created_at: str
    updated_at: str
    message_count: int


class ConversationDetail(BaseModel):
    id: int
    agent_type: str
    title: str | None
    messages: list[dict[str, Any]]


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


def get_conversation_meta_or_404(conversation_id: int) -> tuple[int, str, str | None]:
    """读取会话元数据，找不到时返回 404。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, agent_type, title FROM conversations WHERE id = %s",
                (conversation_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="会话不存在")
    return int(row[0]), row[1], row[2]


def ensure_conversation_owner(conversation_id: int, user_id: int) -> tuple[str, str | None]:
    """确保当前 JWT 用户拥有该会话，防止跨用户读取或追加消息。"""
    owner_id, agent_type, title = get_conversation_meta_or_404(conversation_id)
    if owner_id != user_id:
        raise HTTPException(status_code=403, detail="无权访问该会话")
    return agent_type, title


def sse_event(event_type: str, data: dict) -> str:
    """把数据按 SSE 协议编码为字符串。"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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


def stream_budget_react(
    client: Any,
    messages: list[dict[str, Any]],
    max_tool_rounds: int = 10,
):
    """
    流式版预算 ReAct 循环。

    工具决策轮使用非流式调用，便于稳定读取 tool_calls；
    最终文本回复轮再使用 stream=True，把内容按 SSE text 事件推给前端。
    """
    tool_calls_made: list[str] = []

    for round_num in range(1, max_tool_rounds + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=budget_planner.TOOLS,
            )
        except OpenAIError as exc:
            yield sse_event("error", {"message": f"DeepSeek 调用失败: {exc}"})
            return
        except Exception as exc:
            yield sse_event("error", {"message": f"未知异常: {exc}"})
            return

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if tool_calls:
            append_assistant_with_tool_calls(messages, assistant_message)

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_calls_made.append(tool_name)
                yield sse_event("tool_call", {"name": tool_name, "round": round_num})

                result_text = execute_budget_tool(
                    tool_name,
                    tool_call.function.arguments,
                    messages,
                )
                yield sse_event(
                    "tool_result",
                    {"name": tool_name, "summary": result_text[:120]},
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": result_text,
                    }
                )
            continue

        try:
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=budget_planner.TOOLS,
                stream=True,
            )
        except OpenAIError as exc:
            yield sse_event("error", {"message": f"流式调用失败: {exc}"})
            return
        except Exception as exc:
            yield sse_event("error", {"message": f"流式未知异常: {exc}"})
            return

        full_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            content_piece = getattr(delta, "content", None)
            if content_piece:
                full_text += content_piece
                yield sse_event("text", {"chunk": content_piece})

        messages.append({"role": "assistant", "content": full_text})
        yield sse_event(
            "done",
            {
                "tool_calls_made": tool_calls_made,
                "budget_saved": "save_budget_plan" in tool_calls_made,
            },
        )
        return

    yield sse_event("error", {"message": "达到最大工具循环次数"})


@app.post("/api/agent/anomaly-check", response_model=AnomalyCheckResponse)
def anomaly_check(
    request: AnomalyCheckRequest,
    user_id: int = Depends(get_current_user_id),
) -> AnomalyCheckResponse:
    """异常消费检测：执行一轮 ReAct，解析最终 JSON 并返回前端友好结构。"""
    raw_response = run_anomaly_react(user_id, request.days)
    data = parse_anomaly_response(raw_response)
    return AnomalyCheckResponse(**data)


@app.post("/api/agent/budget/chat", response_model=BudgetChatResponse)
def budget_chat(
    request: BudgetChatRequest,
    user_id: int = Depends(get_current_user_id),
) -> BudgetChatResponse:
    """多轮预算规划：支持 conversation_id 持久化历史，也兼容前端直接传完整历史。"""
    client = get_client_or_raise()

    if request.conversation_id is not None:
        ensure_conversation_owner(request.conversation_id, user_id)
        db_messages = get_conversation_messages(request.conversation_id)
        existing_count = len(db_messages)
        if len(request.messages) > existing_count:
            new_msgs = request.messages[existing_count:]
            messages = db_messages + [message_to_dict(message) for message in new_msgs]
        elif request.messages and request.messages[-1].role == "user":
            latest_message = message_to_dict(request.messages[-1])
            if not db_messages or latest_message.get("content") != db_messages[-1].get("content"):
                messages = db_messages + [latest_message]
            else:
                messages = db_messages
        else:
            messages = db_messages
        conv_id = request.conversation_id
    else:
        messages = [message_to_dict(message) for message in request.messages]
        messages = ensure_budget_system_message(user_id, messages)
        conv_id = create_conversation(
            user_id,
            "budget_planner",
            title="预算规划对话",
        )

    reply, tool_calls_made = run_budget_until_text(client, messages)
    save_messages_batch(conv_id, messages)

    return BudgetChatResponse(
        reply=reply,
        messages=messages,
        tool_calls_made=tool_calls_made,
        budget_saved="save_budget_plan" in tool_calls_made,
        conversation_id=conv_id,
    )


@app.post("/api/agent/budget/chat/stream")
def budget_chat_stream(
    request: BudgetChatRequest,
    user_id: int = Depends(get_current_user_id),
):
    """
    流式版预算对话。返回 SSE 事件流。

    Request 格式与 /api/agent/budget/chat 一致；支持 conversation_id 从 DB 加载历史，
    不传 conversation_id 时自动创建新会话。
    """
    client = get_client_or_raise()

    if request.conversation_id is not None:
        ensure_conversation_owner(request.conversation_id, user_id)
        db_messages = get_conversation_messages(request.conversation_id)
        existing_count = len(db_messages)
        if len(request.messages) > existing_count:
            new_msgs = request.messages[existing_count:]
            messages = db_messages + [message_to_dict(message) for message in new_msgs]
        elif request.messages and request.messages[-1].role == "user":
            latest_message = message_to_dict(request.messages[-1])
            if not db_messages or latest_message.get("content") != db_messages[-1].get("content"):
                messages = db_messages + [latest_message]
            else:
                messages = db_messages
        else:
            messages = db_messages
        conv_id = request.conversation_id
    else:
        messages = [message_to_dict(message) for message in request.messages]
        messages = ensure_budget_system_message(user_id, messages)
        conv_id = create_conversation(
            user_id,
            "budget_planner",
            title="预算规划对话",
        )

    def event_generator():
        yield sse_event("start", {"conversation_id": conv_id})

        for sse_chunk in stream_budget_react(client, messages):
            yield sse_chunk

        save_messages_batch(conv_id, messages)
        yield sse_event(
            "final",
            {
                "messages": messages,
                "conversation_id": conv_id,
            },
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/agent/budget/start", response_model=BudgetStartResponse)
def budget_start(
    user_id: int = Depends(get_current_user_id),
) -> BudgetStartResponse:
    """启动预算规划新对话：创建 system message，并让 Agent 主动开场。"""
    client = get_client_or_raise()
    messages = [{"role": "system", "content": build_system_prompt(user_id)}]
    reply, _ = run_budget_until_text(client, messages)
    conv_id = create_conversation(user_id, "budget_planner", title="预算规划对话")
    save_messages_batch(conv_id, messages)
    return BudgetStartResponse(messages=messages, reply=reply, conversation_id=conv_id)


@app.get("/api/agent/conversations", response_model=list[ConversationSummary])
def list_conversations(
    user_id: int = Depends(get_current_user_id),
    agent_type: str | None = Query(None, description="Agent 类型，可选"),
    limit: int = Query(20, description="最多返回多少条会话"),
) -> list[ConversationSummary]:
    """列出某用户的历史会话，按最近更新时间倒序。"""
    conversations = list_user_conversations(user_id, agent_type, limit)
    return [ConversationSummary(**conversation) for conversation in conversations]


@app.get("/api/agent/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation_detail(
    conversation_id: int,
    user_id: int = Depends(get_current_user_id),
) -> ConversationDetail:
    """获取某会话的完整消息历史。"""
    messages = get_conversation_messages(conversation_id)
    if not messages:
        raise HTTPException(status_code=404, detail="会话不存在或无消息")

    agent_type, title = ensure_conversation_owner(conversation_id, user_id)

    return ConversationDetail(
        id=conversation_id,
        agent_type=agent_type,
        title=title,
        messages=messages,
    )


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", 8001))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
