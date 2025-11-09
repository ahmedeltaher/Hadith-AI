#!/usr/bin/env python3
"""
Main CLI interface for Arabic Hadith RAG Pipeline.
Provides interactive query functionality with various options.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import get_config, update_config
from src.document_loader import HadithDocumentLoader
from src.index_builder import HadithIndexBuilder
from src.query_engine import HadithQueryEngine


app = typer.Typer(
    help="Arabic Hadith RAG Pipeline - Retrieval-Augmented Generation for Hadith texts",
    context_settings={"help_option_names": ["-h", "--help"]}
)
console = Console()


def print_banner():
    """Print application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    Arabic Hadith RAG Pipeline                 ║
    ║                                                              ║
    ║    🕌 Retrieval-Augmented Generation for Hadith Texts       ║
    ║    📚 Powered by LlamaIndex + Ollama + Qdrant              ║
    ║    🤖 Models: qwen2.5:7b + qwen3-embedding:4b              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def print_stats(stats: Dict[str, Any], title: str = "Statistics"):
    """Print statistics in a formatted table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats.items():
        # Format key to be more readable
        formatted_key = key.replace("_", " ").title()
        table.add_row(formatted_key, str(value))
    
    console.print(table)


def print_query_result(result: Dict[str, Any]):
    """Print query result in formatted style."""
    # Print question
    console.print(f"\n❓ [bold cyan]Question:[/bold cyan] {result['question']}")
    
    # Print answer
    if result.get("error"):
        console.print(f"❌ [red]Error:[/red] {result['answer']}")
        return
    
    answer_panel = Panel(
        result["answer"],
        title="🤖 Answer",
        title_align="left",
        border_style="green"
    )
    console.print(answer_panel)
    
    # Print sources if available
    if "sources" in result and result["sources"]:
        console.print("\n📖 [bold]Sources:[/bold]")
        
        for source in result["sources"][:3]:  # Show top 3 sources
            score_text = f" (Score: {source['score']:.3f})" if source.get('score') else ""
            console.print(f"   {source['rank']}.{score_text}")
            
            # Show metadata
            metadata = source.get('metadata', {})
            if metadata:
                console.print(f"      📂 {metadata.get('collection', 'Unknown')} - {metadata.get('book', 'Unknown')}")
            
            # Show text snippet
            console.print(f"      💬 \"{source['text_snippet']}\"")
            console.print()


@app.command()
def interactive(
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild the index from scratch"),
    no_semantic: bool = typer.Option(False, "--no-semantic", help="Disable semantic chunking (use simple chunking)"),
    top_k: int = typer.Option(5, "--top-k", help="Number of documents to retrieve"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Directory containing Hadith documents"),
    storage_dir: Optional[str] = typer.Option(None, "--storage-dir", help="Directory to store index"),
):
    """Start interactive query session with semantic chunking for optimal Hadith boundaries."""
    print_banner()
    
    config = get_config()
    
    # Update config if custom directories provided
    if data_dir:
        update_config(DATA_DIR=Path(data_dir))
    if storage_dir:
        update_config(STORAGE_DIR=Path(storage_dir))
    
    console.print(f"📁 Data Directory: {config.DATA_DIR}")
    console.print(f"💾 Storage Directory: {config.STORAGE_DIR}")
    console.print(f"🔧 Rebuild Index: {rebuild}")
    console.print(f"� Semantic Chunking: {not no_semantic}")
    console.print(f"🔍 Top-K Retrieval: {top_k}")
    console.print()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize components
            task = progress.add_task("Initializing Hadith RAG Pipeline...", total=None)
            
            # Build/load index with semantic chunking
            progress.update(task, description="Building/loading document index with semantic chunking...")
            builder = HadithIndexBuilder(
                storage_dir=config.STORAGE_DIR,
                use_sentence_window=False,  # Always use semantic chunking for better Hadith boundaries
                rebuild=rebuild
            )
            index = builder.build_index(data_dir=config.DATA_DIR)
            
            # Create query engine
            progress.update(task, description="Setting up query engine...")
            query_engine = HadithQueryEngine(
                index=index,
                similarity_top_k=top_k,
                use_reranking=False,  # Can be made configurable
            )
            
            progress.update(task, description="Ready!", completed=True)
        
        console.print("✅ [green]Pipeline initialized successfully![/green]\n")
        
        # Interactive query loop
        console.print("💬 [bold]Enter your questions about Hadith (type 'quit' to exit)[/bold]")
        console.print("Available commands:")
        console.print("  • [cyan]quit[/cyan] - Exit the program")
        console.print("  • [cyan]stats[/cyan] - Show index statistics")
        console.print("  • [cyan]config[/cyan] - Show current configuration")
        console.print("  • [cyan]docs <query>[/cyan] - Show relevant documents without answer")
        console.print()
        
        while True:
            try:
                question = Prompt.ask("🤔 [bold]Your question[/bold]")
                
                if question.lower() in ["quit", "exit", "q"]:
                    console.print("👋 Goodbye!")
                    break
                
                elif question.lower() == "stats":
                    try:
                        stats = builder.get_index_stats(index)
                        print_stats(stats, "Index Statistics")
                    except Exception as e:
                        console.print(f"❌ Error getting stats: {e}")
                    continue
                
                elif question.lower() == "config":
                    config_dict = {
                        "LLM Model": config.LLM_MODEL,
                        "Embedding Model": config.EMBEDDING_MODEL,
                        "Ollama URL": config.OLLAMA_BASE_URL,
                        "Top-K Retrieval": top_k,
                        "Semantic Chunking": not no_semantic,
                        "Collection Name": config.QDRANT_COLLECTION_NAME,
                    }
                    print_stats(config_dict, "Current Configuration")
                    continue
                
                elif question.lower().startswith("docs "):
                    search_query = question[5:].strip()
                    if search_query:
                        console.print(f"🔍 Searching for: {search_query}")
                        docs = query_engine.get_relevant_documents(search_query, top_k)
                        
                        console.print(f"\n📚 Found {len(docs)} relevant documents:")
                        for doc in docs[:5]:  # Show top 5
                            score_text = f" (Score: {doc['score']:.3f})" if doc.get('score') else ""
                            console.print(f"\n{doc['rank']}.{score_text}")
                            
                            metadata = doc.get('metadata', {})
                            if metadata:
                                console.print(f"   📂 {metadata.get('collection', 'Unknown')} - {metadata.get('book', 'Unknown')}")
                            
                            # Truncate long text
                            text = doc['text']
                            if len(text) > 300:
                                text = text[:300] + "..."
                            console.print(f"   💬 {text}")
                    continue
                
                # Regular question
                if question.strip():
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("🤖 Processing your question..."),
                        console=console,
                    ) as progress:
                        task = progress.add_task("Thinking...", total=None)
                        result = query_engine.query(question, include_sources=True)
                        progress.update(task, completed=True)
                    
                    print_query_result(result)
                
            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!")
                break
            except Exception as e:
                console.print(f"❌ [red]Error:[/red] {str(e)}")
    
    except KeyboardInterrupt:
        console.print("\n👋 Interrupted by user")
    except Exception as e:
        console.print(f"❌ [red]Fatal error:[/red] {str(e)}")
        return 1


@app.command()
def build_index(
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Directory containing documents"),
    storage_dir: Optional[str] = typer.Option(None, "--storage-dir", help="Directory to store index"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild existing index"),
    no_semantic: bool = typer.Option(False, "--no-semantic", help="Disable semantic chunking"),
):
    """Build or rebuild the document index with semantic chunking for optimal Hadith boundaries."""
    console.print("🔨 Building Hadith document index...")
    
    config = get_config()
    
    # Update config if directories provided
    if data_dir:
        config.DATA_DIR = Path(data_dir)
    if storage_dir:
        config.STORAGE_DIR = Path(storage_dir)
    
    try:
        # Load documents first to show stats
        loader = HadithDocumentLoader(config.DATA_DIR)
        documents = loader.load_documents()
        
        if not documents:
            console.print("❌ No documents found to index!")
            return 1
        
        doc_stats = loader.get_document_stats(documents)
        print_stats(doc_stats, "Document Statistics")
        
        # Build index with semantic chunking
        builder = HadithIndexBuilder(
            storage_dir=config.STORAGE_DIR,
            use_sentence_window=False,  # Always use semantic chunking for better Hadith boundaries
            rebuild=rebuild
        )
        
        index = builder.build_index(documents=documents)
        index_stats = builder.get_index_stats(index)
        print_stats(index_stats, "Index Statistics")
        
        console.print("✅ [green]Index built successfully![/green]")
        
    except Exception as e:
        console.print(f"❌ [red]Error building index:[/red] {str(e)}")
        return 1


@app.command()
def query_single(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, "--top-k", help="Number of documents to retrieve"),
    show_sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source documents"),
):
    """Execute a single query and exit."""
    console.print(f"❓ Processing question: {question}")
    
    try:
        # Initialize query engine
        query_engine = HadithQueryEngine(similarity_top_k=top_k)
        
        # Execute query
        result = query_engine.query(question, include_sources=show_sources)
        print_query_result(result)
        
    except Exception as e:
        console.print(f"❌ [red]Error:[/red] {str(e)}")
        return 1


@app.command()
def check_setup():
    """Check if Ollama models and dependencies are available."""
    console.print("🔍 Checking system setup...")
    
    config = get_config()
    
    # Check directories
    console.print(f"📁 Data directory: {config.DATA_DIR} {'✅' if config.DATA_DIR.exists() else '❌'}")
    console.print(f"💾 Storage directory: {config.STORAGE_DIR} {'✅' if config.STORAGE_DIR.exists() else '❌'}")
    
    # Check Ollama connection
    try:
        import requests
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            available_models = [model["name"] for model in models.get("models", [])]
            
            console.print(f"🤖 Ollama server: ✅ Connected ({config.OLLAMA_BASE_URL})")
            console.print(f"📋 Available models: {', '.join(available_models)}")
            
            # Check required models
            required_models = [config.LLM_MODEL, config.EMBEDDING_MODEL]
            for model in required_models:
                if model in available_models:
                    console.print(f"   ✅ {model}")
                else:
                    console.print(f"   ❌ {model} (missing - run: ollama pull {model})")
        else:
            console.print(f"🤖 Ollama server: ❌ Error (HTTP {response.status_code})")
    
    except Exception as e:
        console.print(f"🤖 Ollama server: ❌ Connection failed ({e})")
    
    # Check Python dependencies
    console.print("\n📦 Python dependencies:")
    required_packages = ["llama_index", "qdrant_client", "typer", "rich"]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            console.print(f"   ✅ {package}")
        except ImportError:
            console.print(f"   ❌ {package} (run: pip install {package})")


if __name__ == "__main__":
    app()