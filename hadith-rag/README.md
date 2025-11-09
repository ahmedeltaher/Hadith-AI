# Arabic Hadith RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) system specialized for Arabic Hadith texts, built with LlamaIndex, Ollama, and Qdrant vector store.

## 🌟 Features

- **Arabic Hadith Processing**: Specialized pipeline for Arabic religious texts
- **Modern RAG Architecture**: LlamaIndex + Ollama + Qdrant integration
- **Smart Semantic Chunking**: Each Hadith becomes its own node based on semantic meaning boundaries
- **Arabic-Optimized Chunking**: Intelligent document segmentation that preserves Hadith integrity
- **Multiple Data Formats**: Support for .txt, .md, and .json files with automatic boundary detection
- **Interactive CLI**: Rich terminal interface for queries with semantic context
- **Modular Design**: Clean, extensible codebase with smart chunking algorithms

## 🛠️ Technology Stack

- **LLM**: Ollama qwen2.5:7b (Chinese model with Arabic capabilities)
- **Embeddings**: Ollama qwen3-embedding:4b
- **Vector Store**: Qdrant (in-memory or server)
- **Framework**: LlamaIndex ≥0.11.0
- **Interface**: Typer + Rich for beautiful CLI
- **Chunking**: SemanticSplitterNodeParser for optimal Hadith boundaries

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama** (https://ollama.ai)
2. **Pull required models**:
   ```bash
   ollama pull qwen2.5:7b
   ollama pull qwen3-embedding:4b
   ```

### Installation

#### Option 1: Automated Setup
```bash
git clone <repository-url>
cd hadith-rag
./setup.sh
```

#### Option 2: Conda Environment Setup
```bash
git clone <repository-url>
cd hadith-rag

# Create conda environment
conda env create -f environment.yml
conda activate hadith-rag

# Or use make commands
make conda-setup
conda activate hadith-rag
```

#### Option 3: Manual Setup
```bash
git clone <repository-url>
cd hadith-rag

# With pip
pip install -r requirements.txt

# Or with conda
conda create -n hadith-rag python=3.11 -y
conda activate hadith-rag
pip install -r requirements.txt
```

2. **Add your Hadith data** to the `data/` directory:
   - `.json` files with structure: `{"hadiths": [{"text": "...", ...}]}`
   - `.txt` or `.md` files with Hadith texts
   - Sample files included for testing

3. **Run the pipeline**:
   ```bash
   python main.py interactive
   ```

## 📖 Usage

### Interactive Mode (with Semantic Chunking)

Start an interactive query session with smart semantic chunking:

```bash
python main.py interactive [OPTIONS]
```

Options:
- `--rebuild`: Rebuild the index from scratch
- `--no-semantic`: Disable semantic chunking (use simple chunking)
- `--top-k N`: Number of documents to retrieve (default: 5)
- `--data-dir PATH`: Custom data directory
- `--storage-dir PATH`: Custom storage directory

### Single Query

Execute a single query and exit:

```bash
python main.py query-single "ما هو حديث النية؟"
```

### Build Index Only

Build or rebuild the document index with semantic chunking:

```bash
python main.py build-index [OPTIONS]
```

### System Check

Verify setup and dependencies:

```bash
python main.py check-setup
```

## 🧠 Semantic Chunking Features

### Smart Hadith Boundaries
- **Semantic Splitter**: Uses `SemanticSplitterNodeParser` to identify natural boundaries between Hadiths
- **Meaning Preservation**: Each Hadith becomes its own node based on semantic meaning rather than arbitrary character limits
- **Arabic Optimization**: 90% breakpoint threshold optimized for Arabic text patterns
- **Context Awareness**: Maintains relationships between related Hadiths while preserving individual integrity

### Benefits
- ✅ **Better Relevance**: Each node contains complete Hadith context
- ✅ **Improved Accuracy**: No Hadith text is split across multiple nodes
- ✅ **Semantic Understanding**: Boundaries determined by meaning, not length
- ✅ **Enhanced Retrieval**: More precise matching of user queries to relevant Hadiths

## 🗂️ Project Structure

```
hadith-rag/
├── main.py                 # CLI entry point with semantic chunking
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment configuration
├── activate_env.sh         # Environment activation script
├── data/                  # Hadith documents
│   ├── sahih_bukhari_sample.json
│   └── hadith_collection.md
├── storage/               # Vector index storage
├── src/                   # Core modules
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── embeddings.py      # Custom Ollama embedding wrapper
│   ├── document_loader.py # Multi-format document loading
│   ├── index_builder.py   # Semantic chunking & vector index creation
│   └── query_engine.py    # Query processing & response generation
└── README.md
```

## ⚙️ Configuration

Settings are managed in `src/config.py`:

```python
# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "qwen3-embedding:4b"
LLM_MODEL = "qwen2.5:7b"

# Retrieval settings
SIMILARITY_TOP_K = 5

# Semantic Chunking settings  
SEMANTIC_SPLITTER_BREAKPOINT_PERCENTILE_THRESHOLD = 90  # Optimized for Hadith boundaries
```

Environment variables can override defaults via `.env` file.

## 📊 Data Formats

### JSON Format

```json
{
  "collection": "sahih_bukhari",
  "hadiths": [
    {
      "text": "Arabic hadith text here...",
      "english": "English translation",
      "narrator": "Narrator name",
      "grade": "Sahih",
      "number": 1
    }
  ]
}
```

### Text/Markdown Format

```markdown
## Hadith Title

Arabic hadith text here...

## Another Hadith

More Arabic text...
```

The semantic chunking system automatically detects Hadith boundaries based on content structure and meaning.

## 🎯 Example Queries

- "ما هو حديث النية؟" (What is the hadith about intention?)
- "أحاديث عن بر الوالدين" (Hadiths about honoring parents)
- "قال رسول الله عن الصدق" (What the Prophet said about truthfulness)
- "أحاديث في صحيح البخاري عن الصلاة" (Hadiths in Sahih Bukhari about prayer)

## 🔧 Advanced Usage

### Programmatic Usage with Semantic Chunking

```python
from src import HadithQueryEngine, build_hadith_index

# Build index with semantic chunking (default)
index = build_hadith_index(
    data_dir="./data",
    use_sentence_window=False,  # Semantic chunking prioritized
    rebuild=True
)

# Create query engine  
engine = HadithQueryEngine(index, similarity_top_k=10)

# Query
result = engine.query("ما هو الإسلام؟")
print(result["answer"])
```

## 🛡️ Best Practices

1. **Data Quality**: Ensure Arabic text is properly encoded (UTF-8)
2. **Semantic Chunking**: Default behavior provides optimal Hadith boundaries
3. **Index Management**: Use `--rebuild` when adding new documents
4. **Performance**: Semantic chunking may take slightly longer but provides better results
5. **Memory**: Qdrant in-memory mode suitable for smaller datasets

## 🚨 Troubleshooting

### Common Issues

**Semantic Chunking Slow**:
```bash
# Check embedding model is working
python main.py check-setup

# Use simple chunking as fallback
python main.py interactive --no-semantic
```

**Ollama Connection Failed**:
```bash
# Check Ollama is running
ollama serve

# Verify models are available
ollama list
```

**Empty Index**:
```bash
# Rebuild with semantic chunking
python main.py build-index --rebuild
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper documentation
4. Test semantic chunking with sample data
5. Submit a pull request

## 📜 License

This project is open source and available under the MIT License.

---

**Built with ❤️ for Arabic Hadith preservation and accessibility**
**🧠 Enhanced with smart semantic chunking for optimal Hadith boundaries**