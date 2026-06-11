"""
数据库工具函数 - 这些就是 Agent 的 tools

设计原则：
- 每个函数职责单一（一个函数只做一件事）
- 返回结构化数据（dict / list），不返回原始 SQL 结果
- 有清晰的 docstring（LLM 会根据这个描述决定何时调用）
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from statistics import median
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """
    Get a PostgreSQL connection.
    Priority: DATABASE_URL (Railway-style) > individual DB_* env vars (local dev).
    """
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        # Railway/Heroku-style URL.
        # psycopg2 supports postgres:// and postgresql://, but normalize for safety.
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(database_url)

    # Local development: use individual env vars.
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def get_recent_transactions(user_id: int, days: int = 7) -> list:
    """
    查询某用户最近 N 天的交易记录（只查支出，不含收入）
    
    Args:
        user_id: 用户 ID
        days: 查询最近多少天，默认 7 天
    
    Returns:
        交易列表，每条包含 amount, category, merchant, spent_at, note
    """
    sql = """
        SELECT id, amount, category, merchant, spent_at, note
        FROM transactions
        WHERE user_id = %s
          AND type = 'expense'
          AND spent_at >= %s
        ORDER BY spent_at DESC
    """
    since = datetime.now() - timedelta(days=days)
    
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id, since))
            rows = cur.fetchall()
    
    # 把 Decimal 和 datetime 转成普通类型，方便传给 LLM
    return [
        {
            "id": r["id"],
            "amount": float(r["amount"]),
            "category": r["category"],
            "merchant": r["merchant"] or "未知商家",
            "spent_at": r["spent_at"].strftime("%Y-%m-%d %H:%M"),
            "note": r["note"] or "",
        }
        for r in rows
    ]


def get_category_average(user_id: int, category: str, months: int = 3) -> dict:
    """
    查询某用户某分类的历史日均消费（用作异常检测的基线）
    
    Args:
        user_id: 用户 ID
        category: 分类名称，如 "餐饮"、"购物"、"交通"
        months: 统计最近多少个月，默认 3 个月
    
    Returns:
        包含 daily_average（日均）, total_count（总笔数）, period_days（统计天数）
    """
    sql = """
        SELECT 
            COALESCE(SUM(amount), 0) AS total,
            COUNT(*) AS cnt
        FROM transactions
        WHERE user_id = %s
          AND category = %s
          AND type = 'expense'
          AND spent_at >= %s
    """
    since = datetime.now() - timedelta(days=months * 30)
    
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id, category, since))
            row = cur.fetchone()
    
    total = float(row["total"])
    count = int(row["cnt"])
    period_days = months * 30
    daily_avg = total / period_days if period_days > 0 else 0
    
    return {
        "category": category,
        "daily_average": round(daily_avg, 2),
        "total_count": count,
        "total_amount": round(total, 2),
        "period_days": period_days,
    }


def get_category_median_baseline(user_id: int, category: str, days: int = 90) -> dict:
    """
    查询某用户某分类历史单笔消费中位数（用作异常检测基线）

    Args:
        user_id: 用户 ID
        category: 分类名称，如 "餐饮"、"购物"、"交通"
        days: 统计最近多少天，异常检测默认 90 天

    Returns:
        包含 median_amount（单笔中位数）、threshold（2.5 倍阈值）和样本量信息
    """
    sql = """
        SELECT amount
        FROM transactions
        WHERE user_id = %s
          AND category = %s
          AND type = 'expense'
          AND spent_at >= %s
        ORDER BY spent_at DESC
    """
    since = datetime.now() - timedelta(days=days)

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id, category, since))
            rows = cur.fetchall()

    amounts = [float(r["amount"]) for r in rows]
    count = len(amounts)
    has_enough_data = count >= 5

    if not has_enough_data:
        return {
            "category": category,
            "median_amount": 0,
            "threshold": 0,
            "transaction_count": count,
            "baseline_days": days,
            "min_transactions_required": 5,
            "has_enough_data": False,
        }

    median_amount = float(median(amounts))
    threshold = median_amount * 2.5

    return {
        "category": category,
        "median_amount": round(median_amount, 2),
        "threshold": round(threshold, 2),
        "transaction_count": count,
        "baseline_days": days,
        "min_transactions_required": 5,
        "has_enough_data": True,
    }


def get_all_categories(user_id: int) -> list:
    """
    查询某用户所有出现过的消费分类（去重）
    
    Args:
        user_id: 用户 ID
    
    Returns:
        分类名称列表，如 ["餐饮", "购物", "交通"]
    """
    sql = """
        SELECT DISTINCT category
        FROM transactions
        WHERE user_id = %s AND type = 'expense'
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            rows = cur.fetchall()
    
    return [r[0] for r in rows]


# ============== 测试入口 ==============
if __name__ == "__main__":
    """
    直接运行 python db.py 来测试这几个函数是否能连数据库
    """
    USER_ID = 1  # 你的测试用户 test@cmu.edu 的 id
    
    print("=" * 60)
    print("测试 1: 查最近 7 天的交易")
    print("=" * 60)
    txns = get_recent_transactions(USER_ID, days=7)
    print(f"找到 {len(txns)} 条交易")
    for t in txns[:3]:  # 只打印前 3 条
        print(f"  {t['spent_at']}  {t['category']:6s}  ¥{t['amount']:>7.2f}  {t['merchant']}")
    
    print("\n" + "=" * 60)
    print("测试 2: 查餐饮的近 3 个月日均")
    print("=" * 60)
    avg = get_category_average(USER_ID, "餐饮", months=3)
    print(f"  {avg}")
    
    print("\n" + "=" * 60)
    print("测试 3: 查所有出现过的分类")
    print("=" * 60)
    cats = get_all_categories(USER_ID)
    print(f"  {cats}")


def get_monthly_spending_by_category(user_id: int, year_month: str) -> list:
    """
    查询某用户某个月每个消费分类的支出总额。

    当 Agent 需要分析某月预算执行情况、分类消费占比、哪些分类花得最多时，
    调用这个工具获取真实的月度分类支出数据。year_month 格式为 "YYYY-MM"。

    Args:
        user_id: 用户 ID
        year_month: 月份字符串，格式如 "2026-05"

    Returns:
        分类支出列表，每条包含 category, total, count，并按 total 降序排列
    """
    sql = """
        SELECT
            category,
            COALESCE(SUM(amount), 0) AS total,
            COUNT(*) AS cnt
        FROM transactions
        WHERE user_id = %s
          AND type = 'expense'
          AND to_char(spent_at, 'YYYY-MM') = %s
        GROUP BY category
        ORDER BY total DESC
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id, year_month))
            rows = cur.fetchall()

    return [
        {
            "category": r["category"],
            "total": round(float(r["total"]), 2),
            "count": int(r["cnt"]),
        }
        for r in rows
    ]


def get_historical_average_by_category(user_id: int, months: int = 3) -> list:
    """
    查询某用户过去 N 个月每个消费分类的月均消费。

    当 Agent 需要制定预算、评估某分类的合理预算额度、或对比当前月与历史水平时，
    调用这个工具获取分类历史月均消费。计算方式为：最近 months 个月内分类总支出 / months。

    Args:
        user_id: 用户 ID
        months: 回看最近多少个月，默认 3 个月

    Returns:
        分类历史月均列表，每条包含 category, monthly_average, total_in_period
    """
    sql = """
        SELECT
            category,
            COALESCE(SUM(amount), 0) AS total
        FROM transactions
        WHERE user_id = %s
          AND type = 'expense'
          AND spent_at >= %s
        GROUP BY category
        ORDER BY total DESC
    """
    since = datetime.now() - timedelta(days=months * 30)

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id, since))
            rows = cur.fetchall()

    return [
        {
            "category": r["category"],
            "monthly_average": round(float(r["total"]) / months, 2) if months > 0 else 0.0,
            "total_in_period": round(float(r["total"]), 2),
        }
        for r in rows
    ]


def get_current_budgets(user_id: int, year_month: str) -> dict:
    """
    查询某用户某个月已经设置的总预算和分类预算。

    当 Agent 需要查看当前预算、判断是否已有预算方案、或比较实际消费与预算时，
    调用这个工具。budgets 表中 category 为 NULL 表示该月总预算。

    Args:
        user_id: 用户 ID
        year_month: 月份字符串，格式如 "2026-05"

    Returns:
        包含 total_budget 和 category_budgets 的预算字典
    """
    sql = """
        SELECT category, amount
        FROM budgets
        WHERE user_id = %s
          AND year_month = %s
        ORDER BY category NULLS FIRST
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id, year_month))
            rows = cur.fetchall()

    total_budget = None
    category_budgets = []
    for r in rows:
        if r["category"] is None:
            total_budget = float(r["amount"])
        else:
            category_budgets.append(
                {"category": r["category"], "amount": round(float(r["amount"]), 2)}
            )

    return {
        "total_budget": round(total_budget, 2) if total_budget is not None else None,
        "category_budgets": category_budgets,
    }


def save_budget_plan(
    user_id: int,
    year_month: str,
    total_budget: float,
    category_budgets: list,
) -> dict:
    """
    写入或更新某用户某个月的预算方案。

    当 Agent 已经根据收入、历史消费、用户目标生成预算计划，并需要保存到数据库时，
    调用这个工具。total_budget 会保存为 category=NULL 的总预算记录；category_budgets
    会逐条保存为分类预算记录。

    Args:
        user_id: 用户 ID
        year_month: 月份字符串，格式如 "2026-05"
        total_budget: 该月总预算金额
        category_budgets: 分类预算列表，如 [{"category": "餐饮", "amount": 800.0}]

    Returns:
        保存结果，包含 success, saved_count, year_month
    """
    total_budget_sql_delete = """
        DELETE FROM budgets
        WHERE user_id = %s
          AND year_month = %s
          AND category IS NULL
    """
    total_budget_sql_insert = """
        INSERT INTO budgets (user_id, year_month, category, amount, created_at, updated_at)
        VALUES (%s, %s, NULL, %s, NOW(), NOW())
    """
    category_budget_sql = """
        INSERT INTO budgets (user_id, year_month, category, amount, created_at, updated_at)
        VALUES (%s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (user_id, year_month, category)
        DO UPDATE SET
            amount = EXCLUDED.amount,
            updated_at = NOW()
    """

    saved_count = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            # PostgreSQL 的 UNIQUE 约束会把 NULL 当成不相等，所以总预算不能直接依赖
            # ON CONFLICT；先删除旧的 category=NULL 记录，再插入新的总预算记录。
            cur.execute(total_budget_sql_delete, (user_id, year_month))
            cur.execute(total_budget_sql_insert, (user_id, year_month, total_budget))
            saved_count += 1

            for item in category_budgets:
                cur.execute(
                    category_budget_sql,
                    (user_id, year_month, item["category"], item["amount"]),
                )
                saved_count += 1

    return {"success": True, "saved_count": saved_count, "year_month": year_month}


def get_monthly_income(user_id: int, year_month: str) -> float:
    """
    查询某用户某个月的总收入。

    当 Agent 需要制定月度预算、计算可支配收入、或比较收入与支出时，
    调用这个工具获取指定月份的真实收入总额。year_month 格式为 "YYYY-MM"。

    Args:
        user_id: 用户 ID
        year_month: 月份字符串，格式如 "2026-05"

    Returns:
        该月收入总额；没有收入记录时返回 0.0
    """
    sql = """
        SELECT COALESCE(SUM(amount), 0) AS total_income
        FROM transactions
        WHERE user_id = %s
          AND type = 'income'
          AND to_char(spent_at, 'YYYY-MM') = %s
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id, year_month))
            row = cur.fetchone()

    return round(float(row["total_income"]), 2)


def _json_dumps_or_none(value):
    if value is None:
        return None

    import json

    return json.dumps(value, ensure_ascii=False)


def _json_loads_or_none(value):
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value

    import json

    return json.loads(value)


def _to_plain_value(value):
    if value is None:
        return None

    from datetime import date
    from decimal import Decimal

    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def create_conversation(user_id: int, agent_type: str, title: str = None) -> int:
    """
    创建一条新的 Agent 会话，返回新生成的 conversation_id。
    """
    sql = """
        INSERT INTO conversations (user_id, agent_type, title)
        VALUES (%s, %s, %s)
        RETURNING id
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, agent_type, title or "新对话"))
            row = cur.fetchone()

    return int(row[0])


def save_message(
    conversation_id,
    role,
    content,
    tool_calls=None,
    tool_call_id=None,
    tool_name=None,
) -> int:
    """
    保存单条对话消息，并更新会话的 updated_at。
    """
    next_sequence_sql = """
        SELECT COUNT(*) + 1
        FROM conversation_messages
        WHERE conversation_id = %s
    """
    insert_sql = """
        INSERT INTO conversation_messages (
            conversation_id, role, content, tool_calls, tool_call_id, tool_name, sequence_num
        )
        VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
        RETURNING id
    """
    update_sql = """
        UPDATE conversations
        SET updated_at = NOW()
        WHERE id = %s
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(next_sequence_sql, (conversation_id,))
            sequence_num = int(cur.fetchone()[0])
            cur.execute(
                insert_sql,
                (
                    conversation_id,
                    role,
                    content,
                    _json_dumps_or_none(tool_calls),
                    tool_call_id,
                    tool_name,
                    sequence_num,
                ),
            )
            message_id = int(cur.fetchone()[0])
            cur.execute(update_sql, (conversation_id,))

    return message_id


def save_messages_batch(conversation_id, messages: list[dict]) -> int:
    """
    批量保存 OpenAI 格式 messages。已存在的 sequence_num 会自动跳过，避免重复保存。
    """
    current_sequence_sql = """
        SELECT COALESCE(MAX(sequence_num), 0)
        FROM conversation_messages
        WHERE conversation_id = %s
    """
    insert_sql = """
        INSERT INTO conversation_messages (
            conversation_id, role, content, tool_calls, tool_call_id, tool_name, sequence_num
        )
        VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
    """
    update_sql = """
        UPDATE conversations
        SET updated_at = NOW()
        WHERE id = %s
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(current_sequence_sql, (conversation_id,))
            current_sequence = int(cur.fetchone()[0])
            inserted_count = 0

            for sequence_num, message in enumerate(messages, start=1):
                if sequence_num <= current_sequence:
                    continue

                cur.execute(
                    insert_sql,
                    (
                        conversation_id,
                        message.get("role"),
                        message.get("content"),
                        _json_dumps_or_none(message.get("tool_calls")),
                        message.get("tool_call_id"),
                        message.get("name") or message.get("tool_name"),
                        sequence_num,
                    ),
                )
                inserted_count += 1

            if inserted_count > 0:
                cur.execute(update_sql, (conversation_id,))

    return inserted_count


def get_conversation_messages(conversation_id: int) -> list[dict]:
    """
    读取某个会话的全部消息，按 OpenAI messages 格式返回。
    """
    sql = """
        SELECT role, content, tool_calls::text AS tool_calls, tool_call_id, tool_name
        FROM conversation_messages
        WHERE conversation_id = %s
        ORDER BY sequence_num ASC
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (conversation_id,))
            rows = cur.fetchall()

    messages = []
    for row in rows:
        message = {
            "role": row["role"],
            "content": row["content"],
        }

        tool_calls = _json_loads_or_none(row["tool_calls"])
        if tool_calls is not None:
            message["tool_calls"] = tool_calls

        if row["role"] == "tool":
            message["tool_call_id"] = row["tool_call_id"]
            message["name"] = row["tool_name"]

        messages.append(message)

    return messages


def list_user_conversations(user_id: int, agent_type: str = None, limit: int = 20) -> list[dict]:
    """
    列出某用户最近的 Agent 会话，按 updated_at 倒序。
    """
    params = [user_id]
    where_agent_type = ""
    if agent_type is not None:
        where_agent_type = "AND c.agent_type = %s"
        params.append(agent_type)
    params.append(limit)

    sql = f"""
        SELECT
            c.id,
            c.agent_type,
            c.title,
            c.created_at,
            c.updated_at,
            COUNT(m.id) AS message_count
        FROM conversations c
        LEFT JOIN conversation_messages m ON m.conversation_id = c.id
        WHERE c.user_id = %s
          {where_agent_type}
        GROUP BY c.id, c.agent_type, c.title, c.created_at, c.updated_at
        ORDER BY c.updated_at DESC
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return [
        {
            "id": _to_plain_value(row["id"]),
            "agent_type": row["agent_type"],
            "title": row["title"],
            "created_at": _to_plain_value(row["created_at"]),
            "updated_at": _to_plain_value(row["updated_at"]),
            "message_count": int(row["message_count"]),
        }
        for row in rows
    ]


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("测试 4: 创建一个新会话并保存消息")
    print("=" * 60)
    conv_id = create_conversation(USER_ID, "budget_planner", "测试对话")
    print(f"  新会话 ID: {conv_id}")

    save_message(conv_id, "system", "你是预算助手")
    save_message(conv_id, "user", "我想做预算")
    save_message(conv_id, "assistant", "好的，请问哪个月？")
    print(f"  插入 3 条消息后，读取出来：")
    for m in get_conversation_messages(conv_id):
        print(f"    [{m['role']}] {m.get('content', '')[:30]}")

    print("\n" + "=" * 60)
    print("测试 5: 列出用户的会话")
    print("=" * 60)
    convs = list_user_conversations(USER_ID, "budget_planner")
    print(f"  找到 {len(convs)} 个 budget_planner 会话")
    for c in convs[:3]:
        print(f"    #{c['id']} {c['title']} ({c['message_count']} 条消息)")
