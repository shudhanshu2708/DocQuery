# DocQuery 📄

DocQuery is a RAG-based (Retrieval-Augmented Generation) PDF question answering system. Upload a PDF, ask questions in plain language, and get answers grounded in the document, along with the source chunks they came from.

## ✨ Features

- PDF upload with validation (PDF files only)
- Safe handling of empty or unreadable PDFs
- Recursive text chunking (1000 characters, 200 overlap)
- Local embeddings using Ollama (`nomic-embed-text`)
- Persistent vector storage with ChromaDB
- MMR (Maximal Marginal Relevance) retrieval for diverse, relevant context
- Answer generation using Groq (`openai/gpt-oss-20b`)
- Source metadata returned with every answer
- Strict document-only answering: if the answer isn't in the PDF, it replies
  *"I don't know based on the document."*
- Graceful handling of Groq rate limits (HTTP 429)
- Simple web UI
- Interactive API docs via Swagger
- Basic evaluation scripts for testing

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Backend** | Python, FastAPI |
| **RAG** | LangChain |
| **LLM** | Groq (`openai/gpt-oss-20b`) |
| **Embeddings** | Ollama (`nomic-embed-text`) |
| **Vector Database** | ChromaDB |
| **PDF Parsing** | pypdf |
| **Frontend** | HTML, CSS, JavaScript |
| **Config** | python-dotenv |

## 🏗️ Architecture

```text
PDF
 ↓
Text Extraction (pypdf)
 ↓
Chunking (1000 / 200 overlap)
 ↓
Embeddings (Ollama: nomic-embed-text)
 ↓
ChromaDB
 ↓
MMR Retrieval
 ↓
Groq LLM
 ↓
Answer + Sources
```

## 📁 Project Structure

```text
DocQuery/
├── app/
│   ├── main.py
│   ├── core/
│   │   └── vectorstore.py
│   └── routes/
│       └── ask.py
├── evaluation/
│   ├── questions.json
│   └── evaluate.py
├── static/
│   └── index.html
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A [Groq](https://console.groq.com) API key

### Installation

```bash
git clone https://github.com/shudhanshu2708/DocQuery.git
cd DocQuery

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Pull the embedding model

```bash
ollama pull nomic-embed-text
```

### Configure environment variables

Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run the server

```bash
uvicorn app.main:app --reload
```

- Web UI: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/upload` | Upload and index a PDF |
| `POST` | `/ask` | Ask a question, returns answer and sources |
| `GET` | `/count` | Number of chunks stored in the vector database |

### Example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this document?"}'
```

```json
{
  "answer": "...",
  "sources": [
    { "page": 1, "content": "..." }
  ]
}
```

## 🧪 Evaluation

The `evaluation/` folder contains a small test set (`questions.json`) and a script (`evaluate.py`) for checking answer quality during development.

**Note:** The evaluator uses simple word-overlap matching, so it can mark semantically correct answers as wrong. It also hit Groq's daily token limit during testing, which produced HTTP 429 errors. The scores it reports are not a reliable measure of accuracy, and it is kept as a development tool, not a benchmark.

## ⚠️ Limitations

- Answers depend on the quality of text extraction, so scanned PDFs without a text layer aren't supported (no OCR).
- Groq's free tier has daily token limits, so heavy use can return rate-limit errors.
- Ollama must be running locally for embeddings.

## 🔮 Future Improvements

- [ ] OCR support for scanned PDFs
- [ ] Multi-document support
- [ ] Better evaluation using semantic similarity or LLM-based grading
- [ ] Chat history and follow-up questions
- [ ] Docker support
- [ ] React frontend

## 👨‍💻 Author

**Sudhanshu Singh**
GitHub: [@shudhanshu2708](https://github.com/shudhanshu2708)

## 📄 License

This project is licensed under the [MIT License](LICENSE).
