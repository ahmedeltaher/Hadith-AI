# Arabic Hadith RAG Pipeline Makefile
# Provides convenient commands for common operations

.PHONY: help install test check build query clean setup conda-setup conda-install conda-activate

# Default target
help:
	@echo "Arabic Hadith RAG Pipeline"
	@echo "=========================="
	@echo ""
	@echo "Setup commands:"
	@echo "  make setup           - Run initial setup (install deps, download models)"
	@echo "  make conda-setup     - Create and setup conda environment"
	@echo "  make conda-install   - Install dependencies in conda environment"
	@echo "  make install         - Install Python dependencies only"
	@echo ""
	@echo "Usage commands:"
	@echo "  make test            - Run installation tests"
	@echo "  make check           - Check system setup"
	@echo "  make build           - Build/rebuild document index"
	@echo "  make query           - Start interactive query session"
	@echo "  make models          - Download required Ollama models"
	@echo ""
	@echo "Maintenance commands:"
	@echo "  make clean           - Clean storage and cache files"
	@echo "  make conda-clean     - Remove conda environment"
	@echo ""
	@echo "Environment variables:"
	@echo "  DATA_DIR             - Path to Hadith data files (default: ./data)"
	@echo "  STORAGE_DIR          - Path to index storage (default: ./storage)"
	@echo ""

# Run complete setup
setup:
	@echo "🚀 Running complete setup..."
	@./setup.sh

# Install Python dependencies
install:
	@echo "📦 Installing Python dependencies..."
	@pip3 install -r requirements.txt
	@echo "✅ Dependencies installed"

# Run tests
test:
	@echo "🧪 Running installation tests..."
	@python3 test_installation.py

# Check system setup
check:
	@echo "🔍 Checking system setup..."
	@python3 main.py check-setup

# Build document index
build:
	@echo "🔨 Building document index..."
	@python3 main.py build-index $(if $(REBUILD),--rebuild)

# Rebuild index from scratch
rebuild:
	@echo "🔨 Rebuilding document index from scratch..."
	@python3 main.py build-index --rebuild

# Start interactive query session
query:
	@echo "💬 Starting interactive query session..."
	@python3 main.py interactive

# Clean storage and cache
clean:
	@echo "🧹 Cleaning storage and cache files..."
	@rm -rf storage/*
	@rm -rf __pycache__ src/__pycache__
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@echo "✅ Cleaned"

# Download Ollama models
models:
	@echo "📥 Downloading Ollama models..."
	@ollama pull qwen2.5:7b
	@ollama pull qwen3-embedding:4b
	@echo "✅ Models downloaded"

# Development helpers
dev-install:
	@echo "👨‍💻 Installing development dependencies..."
	@pip3 install -r requirements.txt
	@pip3 install pytest black flake8 mypy
	@echo "✅ Development environment ready"

# Format code
format:
	@echo "🎨 Formatting code..."
	@black src/ main.py test_installation.py
	@echo "✅ Code formatted"

# Lint code
lint:
	@echo "🔍 Linting code..."
	@flake8 src/ main.py --max-line-length=88 --ignore=E203,W503
	@echo "✅ Linting complete"

# Run a single query from command line
single-query:
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make single-query Q='Your question here'"; \
		exit 1; \
	fi
	@python3 main.py query-single "$(Q)"

# Show project structure
tree:
	@echo "📂 Project structure:"
	@tree -I '__pycache__|*.pyc|.git' . || find . -type f -not -path './.git/*' -not -path './__pycache__/*' -not -name '*.pyc' | head -20

# Show system information
info:
	@echo "ℹ️  System Information:"
	@echo "Python version: $$(python3 --version)"
	@echo "Pip version: $$(pip3 --version)"
	@echo "Ollama status: $$(ollama --version 2>/dev/null || echo 'Not installed')"
	@echo "Current directory: $$(pwd)"
	@echo "Data directory exists: $$([ -d 'data' ] && echo 'Yes' || echo 'No')"
	@echo "Storage directory exists: $$([ -d 'storage' ] && echo 'Yes' || echo 'No')"

# Conda environment management
conda-setup:
	@echo "🐍 Setting up conda environment..."
	@conda env create -f environment.yml || echo "Environment might already exist"
	@echo "✅ Conda environment 'hadith-rag' ready"
	@echo "To activate: conda activate hadith-rag"

conda-install:
	@echo "📦 Installing dependencies in conda environment..."
	@conda run -n hadith-rag pip install -r requirements.txt
	@echo "✅ Dependencies installed in conda environment"

conda-activate:
	@echo "🔄 To activate the environment, run:"
	@echo "  conda activate hadith-rag"
	@echo "  source activate_env.sh  # For status info"

conda-clean:
	@echo "🗑️ Removing conda environment..."
	@conda env remove -n hadith-rag -y || echo "Environment may not exist"
	@echo "✅ Conda environment removed"

# Export conda environment
conda-export:
	@echo "📄 Exporting conda environment..."
	@conda env export > environment.yml
	@echo "✅ Environment exported to environment.yml"

# Example queries for testing
examples:
	@echo "🎯 Example queries to try:"
	@echo ""
	@echo "Arabic queries:"
	@echo "  make single-query Q='ما هو حديث النية؟'"
	@echo "  make single-query Q='أحاديث عن بر الوالدين'"
	@echo "  make single-query Q='قال رسول الله عن الصدق'"
	@echo ""
	@echo "English queries:"
	@echo "  make single-query Q='What did the Prophet say about intentions?'"
	@echo "  make single-query Q='Hadiths about honoring parents'"
	@echo ""