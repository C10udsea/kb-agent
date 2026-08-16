"""RAG 主流程：检索 -> 拼提示词 -> 生成。

RAG = Retrieval-Augmented Generation（检索增强生成）。
它不重新训练模型，而是把“外部知识”实时检索出来，塞进 prompt，
让 LLM 基于这些资料回答。流程固定为：

问题 -> 向量化 -> 向量库 top-k 检索 -> 原文片段 -> 拼进提示词 -> LLM 回答

请记住计划里那句话：检索质量决定回答质量。
如果 top-k 里没有相关内容，再强的模型也答不对。
"""
from __future__ import annotations

from . import config
from .kb_tools import KnowledgeBase
from .llm import ChatModel, get_chat_model
from .prompts import RAG_SYSTEM_PROMPT


def build_rag_system_prompt(results) -> str:
    """把检索到的原文片段格式化后放进 system prompt。"""
    if not results:
        return RAG_SYSTEM_PROMPT + "\n\n本次没有检索到任何资料。"
    context = "\n\n".join(r.to_prompt_block(i) for i, r in enumerate(results, 1))
    return RAG_SYSTEM_PROMPT + f"\n\n以下是本次从知识库检索到的资料：\n\n{context}"


def ask(
    question: str,
    kb: KnowledgeBase,
    model: ChatModel | None = None,
    top_k: int = 5,
    stream: bool = True,
    show_retrieval: bool = True,
) -> str:
    """单次 RAG 问答：检索 -> 生成 -> 返回答案。"""
    model = model or get_chat_model()
    results = kb.search(question, top_k=top_k)

    if show_retrieval:
        print(f"\n🔍 检索到 {len(results)} 个片段：")
        for r in results:
            print(f"   - [{r.source} · {r.section} · 片段{r.chunk_index}] 相似度 {r.score:.3f}")
        print()

    system = build_rag_system_prompt(results)
    messages = [{"role": "user", "content": question}]

    if stream:
        print("🤖 助手：", end="", flush=True)
        answer = model.stream(
            messages,
            system=system,
            on_text=lambda token: print(token, end="", flush=True),
        )
        print()
        return answer
    resp = model.complete(messages, system=system)
    return resp.text


if __name__ == "__main__":
    # 便于初学者单独查看 prompt 长什么样（不会实际调用模型）
    config.ensure_dirs()
    print(build_rag_system_prompt([]))
