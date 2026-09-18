import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import settings
from backend.api import documents_router, search_router, chat_router
from backend.llm.ollama_client import ollama_client
from backend.database.chroma import db_manager

app = FastAPI(
    title="Clue - Technical Documentation RAG",
    description="Code-Aware Local RAG Application for Technical Documentation using Ollama & ChromaDB",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers
app.include_router(documents_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(chat_router, prefix="/api")

@app.get("/api/health")
async def health_check():
    """System health check verifying Ollama, active models, and vector database status."""
    ollama_health = await ollama_client.check_health()
    docs = db_manager.list_documents()
    collection_count = db_manager.collection.count()

    return {
        "status": "healthy",
        "llm_model": settings.LLM_MODEL,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "ollama": ollama_health,
        "indexed_documents_count": len(docs),
        "total_chunks_count": collection_count
    }

# Mount frontend static directory
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(frontend_dir / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
