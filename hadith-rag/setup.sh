#!/bin/bash

# Arabic Hadith RAG Pipeline Setup Script
# This script helps set up the complete environment

set -e

echo "🕌 Arabic Hadith RAG Pipeline Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+ and try again."
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_status "Python $python_version found"
    
    # Check if version is 3.8+
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_status "Python version is compatible"
    else
        print_error "Python 3.8+ is required. Found version $python_version"
        exit 1
    fi
}

# Check if Ollama is installed
check_ollama() {
    if ! command -v ollama &> /dev/null; then
        print_error "Ollama is not installed."
        echo "Please install Ollama from: https://ollama.ai"
        echo "Then run the following commands:"
        echo "  ollama serve"
        echo "  ollama pull qwen2.5:7b"
        echo "  ollama pull qwen3-embedding:4b"
        exit 1
    fi
    
    print_status "Ollama found"
    
    # Check if Ollama service is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_status "Ollama service is running"
    else
        print_warning "Ollama service may not be running. Start it with: ollama serve"
    fi
}

# Install Python dependencies
install_dependencies() {
    print_step "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_status "Dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Pull Ollama models
setup_ollama_models() {
    print_step "Setting up Ollama models..."
    
    # Check if models exist
    models=$(ollama list 2>/dev/null || echo "")
    
    if echo "$models" | grep -q "qwen2.5:7b"; then
        print_status "qwen2.5:7b model already exists"
    else
        print_step "Downloading qwen2.5:7b model (this may take a while)..."
        ollama pull qwen2.5:7b
        print_status "qwen2.5:7b model downloaded"
    fi
    
    if echo "$models" | grep -q "qwen3-embedding:4b"; then
        print_status "qwen3-embedding:4b model already exists"
    else
        print_step "Downloading qwen3-embedding:4b model..."
        ollama pull qwen3-embedding:4b
        print_status "qwen3-embedding:4b model downloaded"
    fi
}

# Setup directories
setup_directories() {
    print_step "Setting up project directories..."
    
    mkdir -p data storage
    
    if [ ! -f "env.example" ]; then
        print_warning "env.example not found, creating basic configuration..."
        cat > env.example << 'EOF'
# Ollama Settings
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=qwen3-embedding:4b
LLM_MODEL=qwen2.5:7b

# Qdrant Settings
QDRANT_COLLECTION_NAME=hadith_collection
SIMILARITY_TOP_K=5
EOF
    fi
    
    print_status "Project directories ready"
}

# Run system check
run_system_check() {
    print_step "Running system check..."
    
    if python3 main.py check-setup; then
        print_status "System check passed"
    else
        print_warning "System check found issues. Please review the output above."
    fi
}

# Main setup function
main() {
    print_step "Starting Arabic Hadith RAG Pipeline setup..."
    
    # Run checks
    check_python
    check_ollama
    
    # Setup
    install_dependencies
    setup_directories
    setup_ollama_models
    
    # Final check
    run_system_check
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Add your Hadith data files to the 'data/' directory"
    echo "2. Run the pipeline: python3 main.py interactive"
    echo "3. For help: python3 main.py --help"
    echo ""
    echo "Sample commands:"
    echo "  python3 main.py interactive                    # Start interactive mode"
    echo "  python3 main.py build-index --rebuild          # Rebuild index"
    echo "  python3 main.py query-single 'ما هو الإسلام؟'  # Single query"
    echo ""
}

# Handle script interruption
trap 'print_error "Setup interrupted by user"; exit 1' INT

# Run main function
main "$@"