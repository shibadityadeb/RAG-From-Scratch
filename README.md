# RAG with LangChain, LangSmith & Anthropic

A hands-on implementation of Retrieval-Augmented Generation (RAG) built while learning the LangChain ecosystem.

## What is RAG?

RAG combines a retrieval system with a language model. Instead of relying solely on the LLM's training data, it:
1. **Indexes** documents into a vector store
2. **Retrieves** relevant chunks based on the user's query
3. **Generates** an answer by passing the retrieved context to the LLM

## Tech Stack

| Tool | Role |
|------|------|
| **LangChain** | Orchestration framework (chains, retrievers, splitters) |
| **LangSmith** | Tracing and observability for LLM calls |
| **Anthropic Claude** | LLM for answer generation (`claude-3-haiku`) |
| **HuggingFace Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` for vector embeddings |
| **ChromaDB** | Local vector store |
| **PyPDFLoader** | Loading PDF documents |

## Project Structure

```
RAG/
├── data/
│   └── pdf/              # Source PDF documents
├── notebook/
│   └── RAG.ipynb         # Main notebook with full RAG implementation
├── .env                  # API keys (not committed)
├── requirements.txt
└── README.md
```

## What I Learned & Implemented

### Indexing Pipeline
- Load PDFs using `PyPDFLoader` and web pages using `WebBaseLoader`
- Split documents into chunks with `RecursiveCharacterTextSplitter` (chunk_size=300, overlap=50)
- Embed chunks using HuggingFace sentence transformers
- Store embeddings in a local ChromaDB vector store

### Retrieval
- Create a retriever from the vector store with `vectorstore.as_retriever()`
- Retrieve top-k relevant chunks for a query using `retriever.invoke()`

### Generation
- Build a prompt template with `ChatPromptTemplate`
- Chain retriever → prompt → LLM → output parser using LangChain's `|` pipe syntax
- Generate answers grounded in the retrieved context using Claude

### Observability
- Traced all LLM calls via **LangSmith** for debugging and monitoring

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   LANGSMITH_API_KEY=lsv2_...
   ```

3. Add PDF files to `data/pdf/` and open `notebook/RAG.ipynb`

## Key Concepts

- **Chunking**: Large documents are split so each chunk fits within the LLM's context window
- **Embeddings**: Text is converted to numerical vectors to enable semantic similarity search
- **Vector Store**: A database optimized for similarity search over embeddings
- **Retriever**: Finds the most relevant chunks for a given query
- **RAG Chain**: Combines retrieval + generation into a single callable pipeline
