#!/usr/bin/env python3
"""
CropFresh AI Setup Verification Script
======================================
Verifies that all required dependencies and tools are installed correctly.

Usage:
    python scripts/verify_setup.py
"""

import sys
from importlib import import_module
from typing import Any

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
CHECK = "✓"
CROSS = "✗"
WARN = "⚠"


def check_import(module_name: str, package_name: str | None = None) -> tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        mod = import_module(module_name)
        version = getattr(mod, "__version__", "installed")
        return True, version
    except ImportError as e:
        return False, str(e)


def check_groq_connection() -> tuple[bool, str]:
    """Test Groq API connection."""
    try:
        import os
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key or api_key == "your_groq_api_key_here":
            return False, "GROQ_API_KEY not set in .env"
        
        client = Groq(api_key=api_key)
        # Just check if we can list models (minimal API call)
        models = client.models.list()
        return True, f"Connected ({len(list(models.data))} models available)"
    except Exception as e:
        return False, str(e)[:50]


def check_qdrant_connection() -> tuple[bool, str]:
    """Test Qdrant connection."""
    try:
        import os
        from qdrant_client import QdrantClient
        
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        api_key = os.getenv("QDRANT_API_KEY", "")
        
        # Check if using cloud (URL contains qdrant.io or cloud)
        is_cloud = "qdrant.io" in host or "cloud" in host
        
        if is_cloud:
            url = host if host.startswith("https://") else f"https://{host}"
            if ":6333" not in url and ":443" not in url:
                url = f"{url}:6333"
            client = QdrantClient(url=url, api_key=api_key, timeout=10)
        else:
            client = QdrantClient(host=host, port=port, timeout=5)
        
        collections = client.get_collections()
        return True, f"Connected ({len(collections.collections)} collections)"
    except Exception as e:
        if "connect" in str(e).lower():
            return False, f"Cannot connect to {host}:{port}"
        return False, str(e)[:50]


def check_torch_gpu() -> tuple[bool, str]:
    """Check PyTorch and GPU availability."""
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, f"CUDA available: {gpu_name}"
        else:
            return True, "CPU only (no CUDA)"
    except ImportError:
        return False, "PyTorch not installed"


def main():
    print("\n" + "=" * 60)
    print("    🌾 CropFresh AI - Setup Verification")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    # ═══════════════════════════════════════════════════════════════
    # Core Dependencies
    # ═══════════════════════════════════════════════════════════════
    print("📦 Core Dependencies:")
    print("-" * 40)
    
    core_deps = [
        ("langgraph", "LangGraph"),
        ("langchain", "LangChain"),
        ("groq", "Groq SDK"),
        ("qdrant_client", "Qdrant Client"),
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
        ("httpx", "HTTPX"),
    ]
    
    for module, name in core_deps:
        success, info = check_import(module)
        status = f"{GREEN}{CHECK}{RESET}" if success else f"{RED}{CROSS}{RESET}"
        print(f"  {status} {name}: {info}")
        if not success:
            all_passed = False
    
    # ═══════════════════════════════════════════════════════════════
    # ML Dependencies (Optional)
    # ═══════════════════════════════════════════════════════════════
    print("\n🧠 ML Dependencies (Optional):")
    print("-" * 40)
    
    ml_deps = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
    ]
    
    for module, name in ml_deps:
        success, info = check_import(module)
        status = f"{GREEN}{CHECK}{RESET}" if success else f"{YELLOW}{WARN}{RESET}"
        print(f"  {status} {name}: {info}")
    
    # ═══════════════════════════════════════════════════════════════
    # Vision Dependencies (Optional)
    # ═══════════════════════════════════════════════════════════════
    print("\n👁️ Vision Dependencies (Optional):")
    print("-" * 40)
    
    vision_deps = [
        ("ultralytics", "Ultralytics (YOLO)"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
    ]
    
    for module, name in vision_deps:
        success, info = check_import(module)
        status = f"{GREEN}{CHECK}{RESET}" if success else f"{YELLOW}{WARN}{RESET}"
        print(f"  {status} {name}: {info}")
    
    # ═══════════════════════════════════════════════════════════════
    # Voice Dependencies (Optional)
    # ═══════════════════════════════════════════════════════════════
    print("\n🎤 Voice Dependencies (Optional):")
    print("-" * 40)
    
    voice_deps = [
        ("whisper", "OpenAI Whisper"),
    ]
    
    for module, name in voice_deps:
        success, info = check_import(module)
        status = f"{GREEN}{CHECK}{RESET}" if success else f"{YELLOW}{WARN}{RESET}"
        print(f"  {status} {name}: {info}")
    
    # ═══════════════════════════════════════════════════════════════
    # Connection Tests
    # ═══════════════════════════════════════════════════════════════
    print("\n🔌 Connection Tests:")
    print("-" * 40)
    
    # Load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Groq API
    success, info = check_groq_connection()
    status = f"{GREEN}{CHECK}{RESET}" if success else f"{YELLOW}{WARN}{RESET}"
    print(f"  {status} Groq API: {info}")
    
    # Qdrant
    success, info = check_qdrant_connection()
    status = f"{GREEN}{CHECK}{RESET}" if success else f"{YELLOW}{WARN}{RESET}"
    print(f"  {status} Qdrant: {info}")
    
    # GPU
    success, info = check_torch_gpu()
    status = f"{GREEN}{CHECK}{RESET}" if success else f"{YELLOW}{WARN}{RESET}"
    print(f"  {status} GPU: {info}")
    
    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    if all_passed:
        print(f"  {GREEN}All core dependencies verified!{RESET}")
        print("  Run 'uvicorn src.api.main:app --reload' to start the server")
    else:
        print(f"  {YELLOW}Some dependencies missing. Run:{RESET}")
        print("  uv pip install -e '.[all]'")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
