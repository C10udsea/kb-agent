"""OpenAI 兼容 Provider（DeepSeek / GLM）的协议转换测试。

这些测试不调用真实 API，只验证：
1. tool schema 从 Anthropic 形状转成 OpenAI 形状；
2. OpenAI 响应被正确解析成统一的 LLMResponse；
3. Agent 手写循环生成合法的 OpenAI tool_calls / role=tool 消息。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kb_agent.agent import (
    _append_assistant_message,
    _append_tool_results,
    run_agent_manual,
)
from kb_agent.chunking import Chunk
from kb_agent.embeddings import LocalHashEmbedder
from kb_agent.kb_tools import KnowledgeBase, raw_tool_schemas
from kb_agent.llm import (
    ChatModel,
    LLMResponse,
    OpenAICompatChatModel,
    ToolCall,
)
from kb_agent.vector_store import JSONVectorStore


def _fake_openai_client(response):
    chat = SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kwargs: response)
    )
    return SimpleNamespace(chat=chat)


class TestOpenAIToolSchemaConversion(unittest.TestCase):
    def test_anthropic_schema_to_openai_format(self):
        tools = raw_tool_schemas()
        converted = OpenAICompatChatModel._coerce_tools(tools)
        self.assertEqual(converted[0]["type"], "function")
        self.assertEqual(converted[0]["function"]["name"], "search_knowledge_base")
        self.assertEqual(converted[0]["function"]["parameters"], tools[0]["input_schema"])

    def test_system_is_prepended_as_message(self):
        messages = OpenAICompatChatModel._coerce_openai_messages(
            [{"role": "user", "content": "你好"}], "你是一个助手"
        )
        self.assertEqual(messages[0], {"role": "system", "content": "你是一个助手"})
        self.assertEqual(messages[1]["role"], "user")

    def test_tool_message_keeps_tool_call_id(self):
        messages = OpenAICompatChatModel._coerce_openai_messages(
            [{"role": "tool", "tool_call_id": "call_9", "content": "检索结果"}],
            "",
        )
        self.assertEqual(messages[0]["role"], "tool")
        self.assertEqual(messages[0]["tool_call_id"], "call_9")


class TestOpenAICompatChatModel(unittest.TestCase):
    def test_parse_tool_calls(self):
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(
                name="search_knowledge_base",
                arguments='{"query": "过拟合", "top_k": 5}',
            ),
        )
        message = SimpleNamespace(content=None, tool_calls=[tool_call])
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
        )

        model = OpenAICompatChatModel("test-key", "https://example.com", "test-model")
        model.client = _fake_openai_client(response)

        result = model.complete(
            [{"role": "user", "content": "什么是过拟合？"}],
            system="你是助手",
            tools=raw_tool_schemas(),
        )
        self.assertTrue(result.has_tool_calls)
        self.assertEqual(result.tool_calls[0].name, "search_knowledge_base")
        self.assertEqual(result.tool_calls[0].arguments["query"], "过拟合")
        self.assertEqual(result.stop_reason, "tool_calls")

    def test_parse_plain_text(self):
        message = SimpleNamespace(content="你好", tool_calls=None)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")]
        )
        model = OpenAICompatChatModel("test-key", "https://example.com", "test-model")
        model.client = _fake_openai_client(response)
        result = model.complete([{"role": "user", "content": "你好"}])
        self.assertEqual(result.text, "你好")
        self.assertFalse(result.has_tool_calls)


class TestOpenAIAgentLoop(unittest.TestCase):
    def _make_kb(self, tmp: Path) -> KnowledgeBase:
        store = JSONVectorStore(tmp / "store.json")
        embedder = LocalHashEmbedder(dim=64)
        chunks = [Chunk("过拟合是模型记住了训练数据中的噪声。", source="docs/a.md", section="基础", index=0)]
        store.upsert(chunks, embedder.embed_documents([chunks[0].text]), ["a0"])
        return KnowledgeBase(store, embedder)

    def test_manual_loop_uses_openai_wire_messages(self):
        class FakeOpenAIModel(ChatModel):
            name = "fake-openai"
            wire_format = "openai"

            def __init__(self):
                self.turn = 0
                self.sent_messages: list[list[dict]] = []

            def complete(self, messages, system="", tools=None, max_tokens=1024, temperature=0.2):
                self.sent_messages.append([dict(m) for m in messages])
                self.turn += 1
                if self.turn == 1:
                    return LLMResponse(
                        tool_calls=[
                            ToolCall(id="call_1", name="search_knowledge_base", arguments={"query": "过拟合", "top_k": 2})
                        ]
                    )
                return LLMResponse(text="根据检索结果，过拟合是指模型记住了噪声。", stop_reason="stop")

        with tempfile.TemporaryDirectory() as tmp:
            model = FakeOpenAIModel()
            answer = run_agent_manual("什么是过拟合？", self._make_kb(Path(tmp)), model=model, max_turns=3)

        self.assertIn("过拟合", answer)
        # 第二轮请求中的历史：assistant.tool_calls + role=tool 消息
        second_request = model.sent_messages[1]
        self.assertEqual(second_request[1]["role"], "assistant")
        self.assertEqual(second_request[1]["tool_calls"][0]["type"], "function")
        self.assertEqual(second_request[2]["role"], "tool")
        self.assertEqual(second_request[2]["tool_call_id"], "call_1")

    def test_openai_append_helpers(self):
        calls = [ToolCall(id="c1", name="list_documents", arguments={})]
        messages: list[dict] = [{"role": "user", "content": "有哪些文档"}]
        _append_assistant_message(messages, LLMResponse(tool_calls=calls), wire_format="openai")
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["tool_calls"][0]["id"], "c1")

        with tempfile.TemporaryDirectory() as tmp:
            kb = self._make_kb(Path(tmp))
            _append_tool_results(messages, calls, kb, wire_format="openai")
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(messages[-1]["tool_call_id"], "c1")


if __name__ == "__main__":
    unittest.main()
