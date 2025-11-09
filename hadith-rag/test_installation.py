#!/usr/bin/env python3
"""
Test script for Arabic Hadith RAG Pipeline
Validates installation and basic functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test core modules
        from src.config import get_config
        from src.embeddings import OllamaEmbedding
        from src.document_loader import HadithDocumentLoader
        from src.index_builder import HadithIndexBuilder
        from src.query_engine import HadithQueryEngine
        print("✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("🧪 Testing configuration...")
    
    try:
        from src.config import get_config
        config = get_config()
        
        # Check required attributes
        required_attrs = [
            'OLLAMA_BASE_URL', 'EMBEDDING_MODEL', 'LLM_MODEL',
            'DATA_DIR', 'STORAGE_DIR', 'SIMILARITY_TOP_K'
        ]
        
        for attr in required_attrs:
            if not hasattr(config, attr):
                print(f"❌ Missing config attribute: {attr}")
                return False
        
        print("✅ Configuration loaded successfully")
        print(f"   📁 Data directory: {config.DATA_DIR}")
        print(f"   💾 Storage directory: {config.STORAGE_DIR}")
        print(f"   🤖 LLM model: {config.LLM_MODEL}")
        print(f"   🔤 Embedding model: {config.EMBEDDING_MODEL}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_document_loading():
    """Test document loading functionality."""
    print("🧪 Testing document loading...")
    
    try:
        from src.document_loader import HadithDocumentLoader
        from src.config import get_config
        
        config = get_config()
        
        # Check if sample data exists
        if not config.DATA_DIR.exists():
            print(f"❌ Data directory not found: {config.DATA_DIR}")
            return False
        
        # Try to load documents
        loader = HadithDocumentLoader(config.DATA_DIR)
        documents = loader.load_documents()
        
        if documents:
            print(f"✅ Loaded {len(documents)} documents")
            
            # Show first document info
            doc = documents[0]
            print(f"   📄 First document: {len(doc.text)} characters")
            print(f"   🏷️  Metadata keys: {list(doc.metadata.keys())}")
            
            return True
        else:
            print("⚠️  No documents found (this is OK if data/ is empty)")
            return True
            
    except Exception as e:
        print(f"❌ Document loading error: {e}")
        return False


def test_ollama_connection():
    """Test Ollama server connection."""
    print("🧪 Testing Ollama connection...")
    
    try:
        import requests
        from src.config import get_config
        
        config = get_config()
        
        # Test connection
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json()
            available_models = [model["name"] for model in models.get("models", [])]
            
            print("✅ Ollama server connected")
            print(f"   📋 Available models: {len(available_models)}")
            
            # Check required models
            required_models = [config.LLM_MODEL, config.EMBEDDING_MODEL]
            missing_models = []
            
            for model in required_models:
                if model in available_models:
                    print(f"   ✅ {model}")
                else:
                    print(f"   ❌ {model} (missing)")
                    missing_models.append(model)
            
            if missing_models:
                print(f"⚠️  Missing models: {', '.join(missing_models)}")
                print("   Run: ollama pull <model_name>")
                return False
            
            return True
        else:
            print(f"❌ Ollama server error: HTTP {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Ollama connection failed: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return False


def run_all_tests():
    """Run all tests and return overall result."""
    print("🧪 Running Arabic Hadith RAG Pipeline Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config), 
        ("Document Loading", test_document_loading),
        ("Ollama Connection", test_ollama_connection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 20)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Add Hadith data to the data/ directory")
        print("2. Run: python main.py interactive")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)