"""知识库工具：这是阶段 2（Agent）的关键。

Tool Use / Function Calling 的思路：
1. 把“检索知识库”包装成一个函数；
2. 用 JSON Schema 描述这个函数的用途和参数；
3. 把 schema 随请求发给 LLM；
4. LLM 不直接执行函数，而是返回一个 tool_use 请求
   （“我想调用 search_knowledge_base，参数是……”）；
5. 你的程序真正执行函数，把结果以 tool_result 发回；
6. LLM 看到结果后决定继续调用还是给出最终回答。

这里的 search_knowledge_base 和 list_documents 就是给 LLM 使用的工具。
"""
from __future__ import annotations

from .embeddings import Embedder
from .vector_store import SearchResult, VectorStore


class KnowledgeBase:
    """把“embedding + 向量检索”封装成 Agent 可调用的工具。"""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 10))
        query_vector = self.embedder.embed_query(query)
        return self.store.search(query_vector, top_k=top_k)

    def list_documents(self) -> str:
        """列出库中的文档清单，供 LLM 了解知识库有什么。"""
        sources = self.store.sources()
        if not sources:
            return "知识库为空。请先运行 scripts/ingest.py 入库。"
        lines = [f"知识库共有 {self.store.count()} 个片段，来自 {len(sources)} 个文档："]
        for source in sources:
            lines.append(f"- {source}")
        return "\n".join(lines)


def format_tool_result(results: list[SearchResult]) -> str:
    """把检索结果整理成给 LLM 看的 tool_result 文本。"""
    if not results:
        return "没有检索到相关内容。可以尝试换一个更短或更具体的关键词。"
    lines = [f"检索到 {len(results)} 段内容：", ""]
    for i, r in enumerate(results, 1):
        lines.append(r.to_prompt_block(i))
        lines.append("")
    return "\n".join(lines).strip()


def raw_tool_schemas() -> list[dict]:
    """原始 JSON Schema 工具定义。

    真实 Agent 使用 SDK 的 @beta_tool 自动生成 schema；
    保留这个函数是为了在 Demo 模式和手动教学循环中使用，
    也方便你观察 schema 长什么样。
    """
    return [
        {
            "name": "search_knowledge_base",
            "description": "在个人知识库中检索与 query 最相关的原文片段。适合查询概念、事实、观点；一次查询不理想时应更换关键词重试。",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词或自然语言短句，例如：过拟合、Transformer 的注意力机制",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回最相关的片段数量，默认 5，范围 1-10",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "list_documents",
            "description": "列出知识库中所有文档和片段数量。当用户询问知识库包含哪些内容时使用。",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]


def make_beta_tools(kb: KnowledgeBase, verbose: bool = True):
    """用 SDK 的 @beta_tool 装饰器创建工具。

    SDK 会从函数的类型注解自动生成 JSON Schema，并由 ToolRunner
    托管“模型调工具 -> 执行 -> 把结果送回 -> 再问模型”的循环。

    要求 anthropic >= 0.122.0。
    """
    from anthropic.lib.tools import beta_tool

    @beta_tool(
        name="search_knowledge_base",
        description="在个人知识库中检索与 query 最相关的原文片段。检索效果不好时应更换关键词重试。",
    )
    def search_knowledge_base(query: str, top_k: int = 5) -> str:
        """Search the local knowledge base and return relevant excerpts."""
        if verbose:
            print(f"\n🔧 [工具调用] search_knowledge_base(query={query!r}, top_k={top_k})")
        results = kb.search(query, top_k=top_k)
        return format_tool_result(results)

    @beta_tool(
        name="list_documents",
        description="列出知识库中的所有文档。用户询问知识库包含哪些内容时使用。",
    )
    def list_documents() -> str:
        """List all documents in the knowledge base."""
        if verbose:
            print("\n🔧 [工具调用] list_documents()")
        return kb.list_documents()

    return [search_knowledge_base, list_documents]
