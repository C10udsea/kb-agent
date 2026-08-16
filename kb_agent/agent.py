"""Agent 循环：让 LLM 自主决定“查什么、查几次”。

阶段 1 的 RAG 是固定流程：用户问 -> 固定检索一次 -> 生成。
阶段 2 的 Agent 则把“检索”变成模型可以反复调用的工具：

用户问 -> LLM 思考 -> 调用 search_knowledge_base("关键词A")
       -> 查看结果 -> 不满意？再调用 search_knowledge_base("关键词B")
       -> 信息够了 -> 综合成最终回答

本文件提供三种实现：
1. run_agent_real: 生产路径。Anthropic 用 SDK 的 beta tool_runner 托管循环；
   DeepSeek / GLM 等 OpenAI 兼容服务使用手写循环（协议相同，只是 SDK 不同）。
2. run_agent_manual: 强制手写 while 循环，完整展示 tool_use / tool_result 往返；
3. run_agent_demo: 无 API Key 时的演示循环。
"""
from __future__ import annotations

import json
import uuid

from . import config
from .kb_tools import (
    KnowledgeBase,
    format_tool_result,
    make_beta_tools,
    raw_tool_schemas,
)
from .llm import (
    AnthropicChatModel,
    ChatModel,
    DemoChatModel,
    LLMResponse,
    get_chat_model,
)
from .prompts import AGENT_SYSTEM_PROMPT


def execute_tool(name: str, arguments: dict, kb: KnowledgeBase, verbose: bool = True) -> str:
    """真正执行工具。生产环境要在这里做参数校验和错误处理。"""
    if name == "search_knowledge_base":
        query = str(arguments.get("query", "")).strip()
        try:
            top_k = int(arguments.get("top_k", 5))
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 10))
        if not query:
            return "错误：query 参数不能为空。"
        if verbose:
            print(f"\n🔧 [工具调用] search_knowledge_base(query={query!r}, top_k={top_k})")
        results = kb.search(query, top_k=top_k)
        return format_tool_result(results)

    if name == "list_documents":
        if verbose:
            print("\n🔧 [工具调用] list_documents()")
        return kb.list_documents()

    return f"错误：未知工具 {name}"


def _append_assistant_message(messages: list[dict], response: LLMResponse, wire_format: str) -> None:
    """把模型回复（文本 + 工具调用）按服务商协议追加到历史里。

    Anthropic 协议：assistant.content 是块列表，含 tool_use 块；
    OpenAI 协议：assistant.tool_calls 是工具调用数组，content 可为 None。
    两种协议都要求工具调用带 id，后面的工具结果用相同 id 配对。
    """
    if wire_format == "openai":
        message: dict = {"role": "assistant", "content": response.text or None}
        if response.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id or uuid.uuid4().hex[:12],
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in response.tool_calls
            ]
        messages.append(message)
        return

    content: list[dict] = []
    if response.text:
        content.append({"type": "text", "text": response.text})
    for call in response.tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": call.id or uuid.uuid4().hex[:12],
                "name": call.name,
                "input": call.arguments,
            }
        )
    messages.append({"role": "assistant", "content": content})


def _append_tool_results(messages: list[dict], calls: list, kb: KnowledgeBase, wire_format: str) -> None:
    """执行所有工具，并按服务商协议把结果追加回历史。"""
    if wire_format == "openai":
        for call in calls:
            result = execute_tool(call.name, call.arguments, kb, verbose=True)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )
        return

    blocks: list[dict] = []
    for call in calls:
        result = execute_tool(call.name, call.arguments, kb, verbose=True)
        blocks.append(
            {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": result,
            }
        )
    messages.append({"role": "user", "content": blocks})


def _print_final_text(response: LLMResponse) -> None:
    if response.text:
        print("\n🤖 助手：")
        print(response.text)


def run_agent_manual(
    question: str,
    kb: KnowledgeBase,
    model: ChatModel | None = None,
    max_turns: int = 6,
) -> str:
    """手写 Agent 循环：最适合学习 tool_use/tool_result 的往返协议。

    每一步你都能在终端看到“模型想调什么工具、程序返回了什么结果”。
    """
    model = model or get_chat_model()
    messages: list[dict] = [{"role": "user", "content": question}]
    tools = raw_tool_schemas()

    wire_format = getattr(model, "wire_format", "anthropic")

    for _ in range(max_turns):
        response = model.complete(messages, system=AGENT_SYSTEM_PROMPT, tools=tools)
        _append_assistant_message(messages, response, wire_format)

        if response.has_tool_calls:
            _append_tool_results(messages, response.tool_calls, kb, wire_format)
            continue

        _print_final_text(response)
        return response.text

    fallback = "（已达到最大工具调用轮数，请重新提问或减少问题范围。）"
    print("\n🤖 助手：")
    print(fallback)
    return fallback


def _run_agent_anthropic(question: str, kb: KnowledgeBase, max_turns: int) -> str:
    """Anthropic 生产路径：用 SDK 的 beta tool_runner。

    @beta_tool 从函数签名自动生成 JSON Schema；
    client.beta.messages.tool_runner(...) 托管整个循环：
    调用模型 -> 发现 tool_use -> 执行 Python 函数 -> 把结果送回 -> 再调用模型。
    """
    import anthropic

    api_key = config.anthropic_key()
    if not api_key:
        raise RuntimeError("缺少 ANTHROPIC_API_KEY")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        tools = make_beta_tools(kb, verbose=True)
        runner = client.beta.messages.tool_runner(
            model=config.anthropic_model(),
            max_tokens=1200,
            messages=[{"role": "user", "content": question}],
            system=AGENT_SYSTEM_PROMPT,
            tools=tools,
            max_iterations=max_turns,
        )
        final_message = runner.until_done()
    except (AttributeError, TypeError, ImportError):
        print("⚠️  当前 SDK 不支持 beta tool_runner，使用手写循环。")
        return run_agent_manual(
            question,
            kb,
            model=AnthropicChatModel(api_key=api_key),
            max_turns=max_turns,
        )

    texts: list[str] = []
    for block in final_message.content:
        if block.type == "text":
            texts.append(block.text)

    answer = "\n".join(t for t in texts if t)
    print("\n🤖 助手：")
    if answer:
        print(answer)
    else:
        print("（回答为空，请检查控制台日志。）")
    return answer


def run_agent_real(question: str, kb: KnowledgeBase, max_turns: int = 6) -> str:
    """生产路径：按 .env 配置选择服务商。

    - anthropic：SDK beta tool_runner 托管循环；
    - deepseek / glm / openai：使用 OpenAI 兼容协议的手写循环。
    """
    provider = config.resolve_llm_provider()
    if provider == "anthropic":
        return _run_agent_anthropic(question, kb, max_turns=max_turns)
    if provider in ("deepseek", "glm", "openai"):
        return run_agent_manual(
            question,
            kb,
            model=get_chat_model(),
            max_turns=max_turns,
        )
    raise RuntimeError("没有可用的 LLM API Key，无法运行真实 Agent。")


def run_agent_demo(question: str, kb: KnowledgeBase, max_turns: int = 6) -> str:
    """无 API Key 的演示 Agent。

    与 run_agent_manual 共用完全相同的循环，只是模型换成 DemoChatModel。
    这保证你学到的协议是真实的；拿到 Key 后行为不变，只有“大脑”变强。
    """
    return run_agent_manual(
        question,
        kb,
        model=DemoChatModel(),
        max_turns=max_turns,
    )


def run_agent(question: str, kb: KnowledgeBase, max_turns: int = 6) -> str:
    if config.has_llm():
        return run_agent_real(question, kb, max_turns=max_turns)
    return run_agent_demo(question, kb, max_turns=max_turns)
