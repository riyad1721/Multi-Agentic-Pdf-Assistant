# 🤖 Multi-Agentic PDF Assistant (CLI)

An intelligent Retrieval-Augmented Generation (RAG) assistant powered by Groq's blazing-fast LLaMA 3 and multilingual embeddings. This assistant reads a PDF from a public URL, indexes it using vector search, and answers your queries in real-time via CLI.

---

## 🚀 Features

- 📄 Load PDF from URL and convert to knowledge base
- 🔍 Retrieve context using multilingual E5 embeddings
- 🧠 Answer questions using Groq's LLaMA 3 (via `phi.llm.groq`)
- 💬 Continue previous chat sessions using PostgreSQL
- 🌐 Multilingual query support (English, Bangla, etc.)
- ✅ CLI-based interaction with memory and search
- 🔐 Secure config using `.env`

---

## 🧪 Tech Stack

| Component         | Technology                                |
|------------------|-------------------------------------------|
| Embeddings        | [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) |
| Vector DB         | PostgreSQL + pgvector (`PgVector2`)       |
| LLM Backend       | Groq (`llama3-8b-8192`)                   |
| Assistant Engine  | [`phi`](https://github.com/premai-io/phi) |
| CLI Interaction   | [`Typer`](https://typer.tiangolo.com/)   |
| Env Management    | `python-dotenv`                          |

---