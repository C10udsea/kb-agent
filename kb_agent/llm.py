"""LLM 封装：DeepSeek / 智谱 GLM / Claude + 无 Key 演示实现。

需要理解的第一件事：Chat API 是“无状态”的。
每次请求都要把完整对话历史（messages）重新发送，服务器不会替你
记住上一轮说了什么。你看到的“多轮对话”，其实是客户端不断把历史
拼回去的结果。

Messages 三种角色：
- system: 全局规则（你是什么、怎么回答），通常放在最外层参数；
- user: 用户说的话；
- assistant: 模型自己之前说的话（多轮时必须回传）。

流式输出（streaming）：
LLM 不是一次生成整段话，而是一个 token 一个 token 地预测。
SDK 的 text_stream 每收到一个 token 就 yield 一次，因此可以
像打字机一样逐字打印，体验更好。
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from . import config


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

    @property
    def input(self) -> dict:
        return self.arguments


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ChatModel:
    name = "chat"
    # 消息在线路上的格式：
    # - anthropic: assistant 消息使用 content 块 + tool_use，工具结果是 user 消息里的 tool_result
    # - openai: assistant 消息使用 tool_calls 字段，工具结果是 role="tool" 消息
    wire_format = "anthropic"

    def complete(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LLMResponse:
        raise NotImplementedError

    def stream(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        on_text: Callable[[str], None] | None = None,
    ) -> str:
        raise NotImplementedError


class AnthropicChatModel(ChatModel):
    """基于 anthropic SDK 的 Claude 实现。"""

    name = "claude"
    wire_format = "anthropic"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key or config.anthropic_key())
        self.model = model or config.anthropic_model()

    def _params(self, messages, system, max_tokens, temperature, tools=None):
        params: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = tools
        return params

    def complete(self, messages, system="", tools=None, max_tokens=1024, temperature=0.2) -> LLMResponse:
        resp = self.client.messages.create(
            **self._params(messages, system, max_tokens, temperature, tools)
        )
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input or {}),
                    )
                )
        return LLMResponse(
            text="\n".join(p for p in text_parts if p),
            tool_calls=tool_calls,
            stop_reason=getattr(resp, "stop_reason", None),
        )

    def stream(self, messages, system="", max_tokens=1024, temperature=0.2, on_text=None) -> str:
        parts: list[str] = []
        with self.client.messages.stream(
            **self._params(messages, system, max_tokens, temperature)
        ) as stream:
            for text in stream.text_stream:
                parts.append(text)
                if on_text:
                    on_text(text)
        return "".join(parts)


# ---------------------------------------------------------------------------
# OpenAI 兼容实现（DeepSeek / 智谱 GLM / 其他兼容服务）
# ---------------------------------------------------------------------------

class OpenAICompatChatModel(ChatModel):
    """OpenAI Chat Completions 兼容接口。

    国内常用：
    - DeepSeek: base_url=https://api.deepseek.com, model=deepseek-chat
    - 智谱 GLM: base_url=https://open.bigmodel.cn/api/paas/v4, model=glm-4-flash

    与 Anthropic 最大的协议差异：
    - system 提示词是 messages 列表里的第一条消息；
    - 工具调用放在 assistant 消息的 tool_calls 字段中；
    - 工具结果是一条 role="tool" 的消息。
    """

    name = "openai-compatible"
    wire_format = "openai"

    def __init__(self, api_key: str, base_url: str, model: str, provider: str = "openai") -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.provider = provider
        self.name = f"{provider}({model})"

    @staticmethod
    def _coerce_openai_messages(messages: list[dict], system: str) -> list[dict]:
        """把内部消息列表转换成 OpenAI 兼容格式。"""
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "tool":
                item: dict = {"role": "tool", "content": content}
                if msg.get("tool_call_id"):
                    item["tool_call_id"] = msg["tool_call_id"]
                out.append(item)
                continue
            if role == "developer":
                role = "system"
            if role == "system":
                out.append({"role": role, "content": content})
                continue
            if isinstance(content, list):
                texts = [
                    str(b.get("text", ""))
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = "\n".join(texts) if texts else None
            out.append({"role": role, "content": content})
            # assistant 消息可能已经带 OpenAI 风格的 tool_calls
            if isinstance(msg.get("tool_calls"), list) and msg.get("tool_calls"):
                out[-1]["tool_calls"] = msg["tool_calls"]
        return out

    @staticmethod
    def _coerce_tools(tools: list[dict] | None) -> list[dict] | None:
        """把本项目的工具 schema（Anthropic 风格）转成 OpenAI 风格。

        Anthropic: {"name": ..., "description": ..., "input_schema": {...}}
        OpenAI:    {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
        """
        if not tools:
            return None
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                converted.append(tool)
                continue
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                    },
                }
            )
        return converted

    def _common_params(self, messages, system, max_tokens, temperature, tools=None):
        params: dict = {
            "model": self.model,
            "messages": self._coerce_openai_messages(messages, system),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            params["tools"] = self._coerce_tools(tools)
        return params

    def complete(self, messages, system="", tools=None, max_tokens=1024, temperature=0.2) -> LLMResponse:
        resp = self.client.chat.completions.create(
            **self._common_params(messages, system, max_tokens, temperature, tools)
        )
        choice = resp.choices[0]
        message = choice.message
        text = message.content or ""
        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))
        return LLMResponse(text=text.strip(), tool_calls=tool_calls, stop_reason=choice.finish_reason)

    def stream(self, messages, system="", max_tokens=1024, temperature=0.2, on_text=None) -> str:
        params = self._common_params(messages, system, max_tokens, temperature)
        params["stream"] = True
        parts: list[str] = []
        stream = self.client.chat.completions.create(**params)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            text = getattr(delta, "content", None)
            if text:
                parts.append(text)
                if on_text:
                    on_text(text)
        return "".join(parts)


# ---------------------------------------------------------------------------
# 演示实现（无 API Key 时使用）
# ---------------------------------------------------------------------------

_CONTEXT_BLOCK_RE = re.compile(r"【资料(\d+)】来源: ([^\n（]+)（章节: ([^，]*)，片段 (\d+)，相关度 [\d.]+）\n(.*?)(?=\n【资料|\Z)", re.DOTALL)


def _parse_context_blocks(text: str) -> list[dict]:
    blocks: list[dict] = []
    for m in _CONTEXT_BLOCK_RE.finditer(text):
        blocks.append(
            {
                "number": m.group(1),
                "source": m.group(2).strip(),
                "section": m.group(3).strip(),
                "chunk_index": m.group(4),
                "text": m.group(5).strip(),
            }
        )
    return blocks


def _last_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(str(block.get("text", "")))
            if texts:
                return "\n".join(texts)
    return ""


def _tool_result_texts(messages: list[dict]) -> list[str]:
    """从对话历史中提取工具返回的文本（演示模式需要自己读历史）。"""
    results: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                value = block.get("content", "")
                if isinstance(value, str):
                    results.append(value)
                elif isinstance(value, list):
                    results.extend(
                        str(item.get("text", ""))
                        for item in value
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
    return results


def _make_demo_answer(context_blocks: list[dict], question: str) -> str:
    if not context_blocks:
        return (
            "（演示模式）我现在没有配置真实 LLM API，只能做“检索演示”。\n"
            "知识库中暂时没有找到与问题直接相关的内容。\n\n"
            "在 .env 中配置 DEEPSEEK_API_KEY 或 GLM_API_KEY 后，我会基于真实 LLM 综合回答。"
        )

    lines = ["（演示模式回答）根据知识库中检索到的片段，归纳如下：", ""]
    for block in context_blocks[:3]:
        text = block["text"].replace("\n", " ").strip()
        if len(text) > 120:
            text = text[:120] + "…"
        lines.append(f"- {block['source']}：{text}")
    lines.append("")
    lines.append("资料来源：")
    seen: set[str] = set()
    for block in context_blocks:
        cite = f"[{block['source']} · {block['section']} · 片段 {block['chunk_index']}]"
        if cite not in seen:
            lines.append("  " + cite)
            seen.add(cite)
    return "\n".join(lines)


class DemoChatModel(ChatModel):
    """无 Key 演示模型。

    它不是一个真正的语言模型。它的行为规则是：
    1. 普通聊天 -> 提示你配置 API Key；
    2. RAG 场景（system 中有【资料N】）-> 摘录检索片段，模拟“带引用回答”；
    3. Agent 场景（传入了 tools）-> 模拟一到两次工具调用：
       第一次检索整句问题；若问题含“对比/和/与/区别”，第二次检索后半部分，
       然后综合两个工具结果。
    """

    name = "demo"
    wire_format = "anthropic"

    def complete(self, messages, system="", tools=None, max_tokens=1024, temperature=0.2) -> LLMResponse:
        question = _last_user_text(messages) or "你好"
        tool_results = _tool_result_texts(messages)

        if tools:
            tool_names = {t.get("name") for t in tools if isinstance(t, dict)}
            has_search = "search_knowledge_base" in tool_names
            has_list = "list_documents" in tool_names

            if has_list and any(k in question for k in ("有哪些文档", "文档列表", "列出文档", "list")):
                if not tool_results:
                    return LLMResponse(tool_calls=[ToolCall(id="demo_list_1", name="list_documents", arguments={})])

            if has_search:
                if not tool_results:
                    query = self._first_query(question)
                    return LLMResponse(
                        tool_calls=[
                            ToolCall(id="demo_search_1", name="search_knowledge_base", arguments={"query": query, "top_k": 5})
                        ]
                    )
                if len(tool_results) == 1:
                    second = self._second_query(question)
                    if second and second != self._first_query(question):
                        return LLMResponse(
                            tool_calls=[
                                ToolCall(id="demo_search_2", name="search_knowledge_base", arguments={"query": second, "top_k": 5})
                            ]
                        )

        # 没有工具调用 -> 生成最终回答
        context_blocks = _parse_context_blocks(system)
        if not context_blocks:
            for result in tool_results:
                context_blocks.extend(_parse_context_blocks(result))
        if tool_results and not context_blocks:
            # 例如 list_documents 返回的是文档清单，不是【资料】格式，直接展示
            return LLMResponse(
                text="（演示模式回答）\n\n" + tool_results[-1],
                stop_reason="end_turn",
            )
        return LLMResponse(text=_make_demo_answer(context_blocks, question), stop_reason="end_turn")

    @classmethod
    def _comparison_parts(cls, question: str) -> tuple[str, str, str]:
        """把“对比 A 和 B 关于 X 的观点”拆成 (A, B, X)。

        演示模型用这条规则模仿真实 LLM 的多步决策：
        第一轮检索 A + X，第二轮检索 B + X。
        """
        q = question.strip().lstrip("?？ ").strip()
        for prefix in ("请对比", "请比较", "对比", "比较"):
            if q.startswith(prefix):
                q = q[len(prefix):].lstrip("：:，, ").strip()
                break
        left = right = topic = ""
        for sep in ("和", "与", "跟", "及", " vs ", " VS "):
            if sep in q:
                left, right = q.split(sep, 1)
                break
        if not left or not right:
            return "", "", ""
        left = left.strip().strip("，,。 ")
        right_raw = right.strip().strip("，,。 ")
        m = re.search(r"关于\s*([^，,。？?的]+)(?:的观点|的区别|的不同|的异同|的差异)?", right_raw)
        if m:
            topic = m.group(1).strip()
        if "关于" in right_raw:
            right = right_raw.split("关于", 1)[0].strip()
        else:
            right = right_raw
        for suffix in ("的观点", "的区别", "的不同", "的异同", "的差异", "有什么不同", "有何不同"):
            if right.endswith(suffix):
                right = right[: -len(suffix)].strip()
        for word in ("里", "中", "里面"):
            right = right.removeprefix(word)
            right = right.removesuffix(word)
        return left, right, topic

    @classmethod
    def _first_query(cls, question: str) -> str:
        left, _, topic = cls._comparison_parts(question)
        if left:
            return f"{left} {topic}".strip()
        return question

    @classmethod
    def _second_query(cls, question: str) -> str:
        _, right, topic = cls._comparison_parts(question)
        if right:
            return f"{right} {topic}".strip()
        return ""

    def stream(self, messages, system="", max_tokens=1024, temperature=0.2, on_text=None) -> str:
        resp = self.complete(messages, system=system, max_tokens=max_tokens, temperature=temperature)
        # 模拟流式：按短句逐步输出，让你在没有 Key 时也能看到“打字机效果”
        piece = ""
        for ch in resp.text:
            piece += ch
            if ch in "。！？\n" or len(piece) >= 6:
                if on_text:
                    on_text(piece)
                piece = ""
        if piece and on_text:
            on_text(piece)
        return resp.text


def get_chat_model() -> ChatModel:
    """按 .env 配置创建 LLM 客户端；什么都没配时退回演示模型。"""
    provider = config.resolve_llm_provider()
    if provider == "anthropic":
        return AnthropicChatModel(api_key=config.anthropic_key())
    if provider in ("deepseek", "glm", "openai"):
        settings = config.llm_settings(provider)
        return OpenAICompatChatModel(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            model=settings["model"],
            provider=provider,
        )
    print("⚠️  未检测到可用的 LLM API Key（支持 DEEPSEEK_API_KEY / GLM_API_KEY），将使用 DemoChatModel。")
    return DemoChatModel()
