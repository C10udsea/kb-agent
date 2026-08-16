# kb-agent

个人知识库问答系统：把 Markdown / TXT 笔记变成可检索、可引用、可多步推理的知识库 Agent。

[![Python](https://img.shields.io/badge/Python-3.10%20%E2%80%93%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-7C3AED)](LICENSE)
[![RAG + Agent](https://img.shields.io/badge/RAG%20%2B%20Agent-DeepSeek%20%2F%20GLM-2F6FED)](#)

kb-agent 不依赖 LangChain 等重框架，从切块、向量检索到 Tool Use 循环全部手写实现。它支持：

- **RAG 问答**：检索最相关的知识片段，回答附引用来源，降低幻觉；
- **多步 Agent**：LLM 自主决定“查什么、查几次”，支持对比类问题的多轮查证；
- **多 Provider**：对话模型支持 DeepSeek / 智谱 GLM / Anthropic / OpenAI 兼容接口，Embedding 支持智谱 GLM / Voyage / OpenAI 兼容接口；
- **无 Key 演示模式**：没有 API Key 时也能完整跑通数据流，便于快速体验；
- **检索质量评估**：内置 Hit@k / MRR 指标，对比不同切块参数与 top-k 的效果。

---

## 架构

```text
Markdown / TXT 文档
        │
        ▼
  切块（chunking）            # 按标题/段落切分，带 overlap
        │
        ▼
  Embedding（GLM / Voyage）   # 文本 → 向量
        │
        ▼
  ChromaDB 向量库             # 持久化存储与余弦检索
        │
        ├── RAG：固定检索一次 → 拼接上下文 → LLM 回答（附引用）
        │
        └── Agent：LLM 调用 search_knowledge_base / list_documents
                     → 多轮检索 → 信息充足后综合回答
```

国内推荐组合：**DeepSeek（对话）+ 智谱 GLM（Embedding）**。

---

## 快速开始

### 1. 环境要求

- Python 3.10 - 3.12（ChromaDB 对新版本 Python 的适配通常慢半拍）
- DeepSeek / 智谱 GLM 的 API Key（可选，没有则进入演示模式）

### 2. 安装

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

或手动创建环境：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux / macOS：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 配置

```powershell
Copy-Item .env.example .env
notepad .env
```

国内推荐配置：

```ini
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的DeepSeekKey

GLM_API_KEY=你的GLMKey
EMBEDDING_PROVIDER=glm
GLM_EMBEDDING_MODEL=embedding-3
```

> `.env` 已被 `.gitignore` 排除，**永远不要**将其提交到 Git。

### 4. 使用

```powershell
# ① 文档入库（切块 -> Embedding -> ChromaDB）
python scripts/ingest.py

# ② RAG 问答（带引用来源）
python scripts/chat.py
python scripts/chat.py -q "香农公式的含义是什么？"

# ③ Agent 问答（多步检索、函数调用）
python scripts/agent.py
python scripts/agent.py -q "对比 OFDM 和 MIMO 的作用"

# ④ 检索质量评估
python scripts/eval.py

# ⑤ 多轮聊天（流式输出）
python scripts/chat_cli.py
```

### 5. 运行测试

```bash
python -m unittest discover -s tests -v
```

---

## Provider 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `deepseek` / `glm` / `anthropic` / `openai` / `auto` |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek 接口地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 对话模型 |
| `GLM_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4` | 智谱接口地址 |
| `GLM_MODEL` | `glm-4-flash` | 智谱对话模型 |
| `EMBEDDING_PROVIDER` | `auto` | `glm` / `voyage` / `openai` / `auto` |
| `GLM_EMBEDDING_MODEL` | `embedding-3` | 如不可用改 `embedding-2` |


---

## 命令行参数

### `ingest.py`

```powershell
python scripts/ingest.py --chunk-size 600 --chunk-overlap 100
python scripts/ingest.py --clear          # 清空后重新入库
```

### `chat.py`

```powershell
python scripts/chat.py --top-k 8
python scripts/chat.py -q "香农公式的含义是什么？"
```

交互命令：`/topk N`、`/sources`、`/exit`。

### `agent.py`

```powershell
python scripts/agent.py -q "对比 OFDM 和 MIMO 的作用"
python scripts/agent.py --max-turns 8
python scripts/agent.py --manual          # 观察 tool_use/tool_result 往返
```

### `eval.py`

```powershell
python scripts/eval.py --chunk-sizes 400,800,1200 --top-k 3,5,8
```

---

## 目录结构

```text
kb-agent/
├── README.md                # 项目说明
├── LICENSE                  # MIT 许可证
├── requirements.txt         # 核心依赖
├── requirements-optional.txt# Anthropic / Voyage 可选依赖
├── .env.example             # 环境变量模板（复制为 .env）
├── docs/                    # 示例知识库与评估问题
│   ├── *.md                 # 通信工程示例笔记（可替换为自己的文档）
│   └── eval_questions.json  # 检索评估问题集
├── kb_agent/                # 核心库
│   ├── config.py            # Provider 选择、路径与 Key 读取
│   ├── chunking.py          # 文档读取与 Markdown 感知切块
│   ├── embeddings.py        # GLM / Voyage / OpenAI / 本地 Embedding
│   ├── vector_store.py      # ChromaDB 封装（含 JSON 后备实现）
│   ├── llm.py               # 多 Provider LLM 封装与协议转换
│   ├── prompts.py           # RAG 与 Agent 提示词
│   ├── kb_tools.py          # search / list_documents 工具
│   ├── rag.py               # RAG 固定流程
│   └── agent.py             # Tool Use Agent 循环
├── scripts/                 # 命令行入口与 Windows 初始化脚本
└── tests/                   # 单元测试
```

---

## 评估

`scripts/eval.py` 使用 `docs/eval_questions.json` 中的问题集，对比不同 `chunk_size` 与 `top-k` 组合的检索效果：

- **Hit@k**：标准答案来源文档是否出现在前 k 个检索结果中；
- **MRR**：第一个正确来源出现位置的倒数平均。

评估使用临时向量库，不会污染你的 `data/` 目录。



---

## License

本项目基于 [MIT License](LICENSE) 开源。
