import sys
import time
import httpx
import uvicorn
from pathlib import Path
from backend.config import settings

def print_banner():
    banner = """
  Production-Oriented Local Code-Aware RAG for Technical Documentation
  LLM: Ollama (model: {llm_model})
  Embedding: {emb_provider} (model: {emb_model})
  Web UI: http://localhost:{port}
========================================================================
""".format(
        llm_model=settings.LLM_MODEL,
        emb_provider=settings.EMBEDDING_PROVIDER,
        emb_model=settings.EMBEDDING_MODEL,
        port=settings.PORT
    )
    print(banner)

def verify_ollama():
    print(f"[*] Checking Ollama status at {settings.OLLAMA_BASE_URL}...")
    try:
        with httpx.Client(timeout=3.0) as client:
            res = client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if res.status_code == 200:
                models = [m["name"] for m in res.json().get("models", [])]
                print(f"[+] Ollama is ONLINE! Available models: {', '.join(models) if models else 'None'}")
                if any(settings.LLM_MODEL in m for m in models):
                    print(f"[+] Configured LLM '{settings.LLM_MODEL}' is ready.")
                else:
                    print(f"[!] WARNING: Model '{settings.LLM_MODEL}' was not found in Ollama.")
                    print(f"    Run: 'ollama pull {settings.LLM_MODEL}' to download it.")

                if settings.EMBEDDING_PROVIDER == "ollama":
                    if any(settings.EMBEDDING_MODEL in m for m in models):
                        print(f"[+] Configured Embedding model '{settings.EMBEDDING_MODEL}' is ready.")
                    else:
                        print(f"[!] Notice: Embedding model '{settings.EMBEDDING_MODEL}' is not yet pulled.")
                        print(f"    You can run: 'ollama pull {settings.EMBEDDING_MODEL}'.")
                        print(f"    (The system will automatically use high-performance local fallback if needed).")
                return True
    except Exception as e:
        print("\n[!] CRITICAL WARNING: Ollama appears to be OFFLINE or unreachable!")
        print(f"    Target URL: {settings.OLLAMA_BASE_URL}")
        print("    Please start the Ollama application or run 'ollama serve' in another terminal.\n")
        return False

def main():
    print_banner()
    verify_ollama()
    print(f"\n[*] Starting Technical RAG Server on http://{settings.HOST}:{settings.PORT} ...")
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)

if __name__ == "__main__":
    main()
