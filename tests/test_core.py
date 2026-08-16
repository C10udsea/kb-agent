"""核心逻辑的单元测试。

运行：.venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kb_agent.chunking import chunk_id, read_documents, split_text
from kb_agent.embeddings import LocalHashEmbedder
from kb_agent.kb_tools import raw_tool_schemas
from kb_agent.llm import DemoChatModel
from kb_agent.rag import build_rag_system_prompt
from kb_agent.vector_store import JSONVectorStore, SearchResult


class TestChunking(unittest.TestCase):
    def test_splits_long_text_and_keeps_metadata(self):
        text = "# 第一章\n\n" + ("过拟合是指模型记住了噪声。" * 80)
        chunks = split_text(text, source="docs/test.md", chunk_size=200, chunk_overlap=40)
        self.assertGreater(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 200)
            self.assertEqual(chunk.source, "docs/test.md")
            self.assertIn("第一章", chunk.section)

    def test_overlap_keeps_tail_of_previous_chunk(self):
        text = ("第一段" + "A" * 180 + "\n\n" + "第二段" + "B" * 180)
        chunks = split_text(text, source="x.md", chunk_size=200, chunk_overlap=30)
        self.assertGreaterEqual(len(chunks), 2)
        # 第二块开头应包含第一块末尾的一部分（允许分段拼接带来的少量偏差）
        self.assertTrue(chunks[1].text.startswith("A") or "A" in chunks[1].text[:80])

    def test_chunk_id_is_stable(self):
        a = chunk_id("docs/a.md", 0, 800, 120)
        b = chunk_id("docs/a.md", 0, 800, 120)
        c = chunk_id("docs/a.md", 0, 500, 120)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_read_documents_ignores_hidden_and_other_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text("内容A", encoding="utf-8")
            (root / "b.txt").write_text("内容B", encoding="utf-8")
            (root / "c.pdf").write_text("忽略", encoding="utf-8")
            (root / ".hidden.md").write_text("忽略", encoding="utf-8")
            docs = read_documents(root)
            self.assertEqual({Path(d.source).name for d in docs}, {"a.md", "b.txt"})


class TestLocalHashEmbedder(unittest.TestCase):
    def setUp(self):
        self.embedder = LocalHashEmbedder(dim=128)

    def test_shape_and_normalization(self):
        vec = self.embedder.embed_query("过拟合")
        self.assertEqual(len(vec), 128)
        norm = sum(x * x for x in vec) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_same_text_similarity_is_one(self):
        a = self.embedder.embed_query("注意力机制")
        b = self.embedder.embed_query("注意力机制")
        self.assertAlmostEqual(sum(x * y for x, y in zip(a, b)), 1.0, places=6)


class TestJSONVectorStore(unittest.TestCase):
    def test_upsert_and_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JSONVectorStore(Path(tmp) / "store.json")
            embedder = LocalHashEmbedder(dim=64)
            from kb_agent.chunking import Chunk

            chunks = [
                Chunk("过拟合是模型记住了噪声。", source="docs/a.md", section="基础", index=0),
                Chunk("向量数据库使用余弦相似度。", source="docs/b.md", section="RAG", index=0),
            ]
            vectors = embedder.embed_documents([c.text for c in chunks])
            store.upsert(chunks, vectors, ["id_a", "id_b"])
            self.assertEqual(store.count(), 2)

            results = store.search(embedder.embed_query("什么是过拟合"), top_k=1)
            self.assertEqual(results[0].source, "docs/a.md")
            self.assertGreater(results[0].score, 0.0)

    def test_source_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JSONVectorStore(Path(tmp) / "store.json")
            embedder = LocalHashEmbedder(dim=64)
            from kb_agent.chunking import Chunk

            chunks = [Chunk("内容一", source="docs/a.md", section="s", index=0)]
            store.upsert(chunks, embedder.embed_documents(["内容一"]), ["1"])
            self.assertEqual(store.search(embedder.embed_query("内容一"), source="docs/b.md"), [])
            self.assertEqual(len(store.search(embedder.embed_query("内容一"), source="docs/a.md")), 1)


class TestDemoAgentBehavior(unittest.TestCase):
    def test_comparison_question_makes_two_different_searches(self):
        model = DemoChatModel()
        question = "对比机器学习和RAG里关于切块的观点"
        tools = raw_tool_schemas()
        first = model.complete([{"role": "user", "content": question}], system="", tools=tools)
        self.assertTrue(first.has_tool_calls)
        self.assertEqual(first.tool_calls[0].name, "search_knowledge_base")
        self.assertIn("机器学习", first.tool_calls[0].arguments["query"])

        # 模拟第一轮 tool_result 返回后，模型应发起第二次检索
        messages = [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": first.tool_calls[0].id,
                        "name": first.tool_calls[0].name,
                        "input": first.tool_calls[0].arguments,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": first.tool_calls[0].id, "content": "检索到 0 段内容。"}
                ],
            },
        ]
        second = model.complete(messages, system="", tools=tools)
        self.assertTrue(second.has_tool_calls)
        self.assertIn("RAG", second.tool_calls[0].arguments["query"])


class TestRagPrompt(unittest.TestCase):
    def test_context_is_embedded_with_citations(self):
        result = SearchResult(
            text="原文内容",
            source="docs/a.md",
            section="第一章",
            chunk_index=0,
            score=0.9,
        )
        prompt = build_rag_system_prompt([result])
        self.assertIn("【资料1】", prompt)
        self.assertIn("docs/a.md", prompt)
        self.assertIn("原文内容", prompt)


if __name__ == "__main__":
    unittest.main()
