# 📚 RAG Document QA System

A production-ready Retrieval-Augmented Generation (RAG) system for document question answering using LangChain, FAISS, and local LLMs.

## 🚀 Features

- **Multiple Document Support**: PDF and TXT files
- **Vector Embeddings**: Using sentence-transformers for semantic search
- **Local LLM Integration**: Works with Ollama (Llama2, Mistral, etc.)
- **Interactive UI**: Streamlit-based web interface
- **Source Citation**: Returns relevant document chunks with answers
- **Customizable**: Adjustable chunk size, temperature, and retrieval parameters

## 🛠️ Installation

### Prerequisites

1. Install Ollama:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

2. Pull a model:
```bash
ollama pull llama2
# or
ollama pull mistral
```

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-document-qa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

### Programmatic Usage

```python
from app import RAGDocumentQA

# Initialize system
rag = RAGDocumentQA(model_name="llama2")

# Load and process documents
documents = rag.load_documents("./docs")
chunks = rag.split_documents(documents)
rag.create_vectorstore(chunks)

# Setup QA chain
rag.setup_qa_chain(temperature=0.7, top_k=4)

# Ask questions
result = rag.ask_question("What is this document about?")
print(result["answer"])
```

## 📁 Project Structure

```
rag-document-qa/
├── app.py                  # Main application
├── requirements.txt        # Dependencies
├── README.md              # Documentation
├── .env.example           # Environment variables template
├── vectorstore/           # FAISS vector store (generated)
└── uploaded_docs/         # Uploaded documents (generated)
```

## ⚙️ Configuration

Adjust these parameters in the UI sidebar:
- **LLM Model**: Choose from available Ollama models
- **Temperature**: Control randomness (0.0 - 1.0)
- **Top K**: Number of relevant documents to retrieve
- **Chunk Size**: Size of document chunks for processing

## 🧪 Example Use Cases

- **Research Papers**: Answer questions about academic papers
- **Legal Documents**: Query contracts and legal texts
- **Technical Documentation**: Search through manuals and guides
- **Business Reports**: Extract insights from reports

## 🔧 Advanced Features

### Custom Embeddings

```python
from langchain_community.embeddings import OpenAIEmbeddings

rag = RAGDocumentQA(
    model_name="llama2",
    embedding_model="BAAI/bge-large-en-v1.5"
)
```

### Persistent Vector Store

```python
# Save
rag.create_vectorstore(chunks, persist_directory="./my_vectorstore")

# Load
rag.load_vectorstore(persist_directory="./my_vectorstore")
```

## 📊 Performance Tips

1. **Chunk Size**: Smaller chunks (500-1000) for precise answers, larger (1500-2000) for context
2. **Overlap**: 200-300 token overlap helps maintain context
3. **Top K**: Start with 3-5 documents, adjust based on results
4. **Model Selection**: Llama2 for general use, Mistral for faster responses

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License

## 🙏 Acknowledgments

- LangChain for the RAG framework
- Ollama for local LLM support
- Sentence Transformers for embeddings
- FAISS for vector similarity search

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.
