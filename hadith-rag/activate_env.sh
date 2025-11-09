#!/bin/bash

# Activation script for Arabic Hadith RAG Pipeline conda environment
# Usage: source activate_env.sh

echo "🕌 Activating Arabic Hadith RAG Pipeline Environment"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    return 1
fi

# Activate the hadith-rag environment
if conda info --envs | grep -q "hadith-rag"; then
    conda activate hadith-rag
    echo "✅ Environment 'hadith-rag' activated"
    
    # Display environment info
    echo "📊 Environment Information:"
    echo "   Python: $(python --version)"
    echo "   Location: $(which python)"
    echo "   Conda Env: $CONDA_DEFAULT_ENV"
    
    # Quick status check
    echo ""
    echo "🔧 Quick Status Check:"
    python -c "
import sys
try:
    from src.config import get_config
    print('   ✅ Configuration module loaded')
except Exception as e:
    print(f'   ❌ Configuration error: {e}')
    
try:
    import llama_index
    print(f'   ✅ LlamaIndex version: {llama_index.__version__}')
except Exception as e:
    print(f'   ❌ LlamaIndex error: {e}')
    
try:
    import qdrant_client
    print('   ✅ Qdrant client available')
except Exception as e:
    print(f'   ❌ Qdrant error: {e}')
    
try:
    import ollama
    print('   ✅ Ollama client available')
except Exception as e:
    print(f'   ❌ Ollama error: {e}')
"
    
    echo ""
    echo "🚀 Ready to use! Available commands:"
    echo "   python main.py interactive     # Start interactive session"
    echo "   python main.py check-setup     # Verify system setup"
    echo "   python main.py build-index     # Build document index"
    echo "   python test_installation.py   # Run installation tests"
    echo ""
    echo "📖 For full help: python main.py --help"
    
else
    echo "❌ Environment 'hadith-rag' not found"
    echo "Please create it first with:"
    echo "  conda env create -f environment.yml"
    echo "or"
    echo "  ./setup.sh"
    return 1
fi