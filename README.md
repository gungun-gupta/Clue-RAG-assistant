# Technical Documentation RAG (Code-Aware Local Assistant)

A production-oriented, local **Retrieval-Augmented Generation (RAG)** application built specifically for **technical documentation** using **Ollama (`llama3.2:3b`)**, **ChromaDB**, and **FastAPI**.

Unlike generic toy RAGs that blindly split text into arbitrary fixed-character slices and lose functional structure, this system is **code-aware**: it analyzes code completeness, detects functional boundaries, and **prioritizes complete production-ready implementations over isolated snippets** when you ask for full code.

---

## 1. Core Architecture: Why Each Component Exists

```
                                [Technical Document]
                            (PDF, DOCX, TXT, MD, Code)
                                         │
                             [Multi-Format Parser]
                       (Extracts pages, text, structure)
                                         │
                         [Code & Completeness Detector]
                      (Calculates score 0.0-1.0 via signals:
                    imports, config, functions, initialization)
                                         │
                            [Code-Aware Chunker]
                     (Preserves functions; parent/child links)
                                         │
                             [ChromaDB Vector Store]
                                         │
   User Query ──► [Query Classifier] ────┤ (COMPLETE_CODE, CODE, EXPLANATION, CONFIG)
                        │                │
                 [Query Expander] ───────┤ (Generates technical search variations)
                        │                │
                 [Hybrid Search] ────────┤ (Vector Semantic + BM25 Keywords)
                        │                │
                 [Code-Aware Reranker] ──┘ (Boosts complete_code, penalizes snippets)
                        │
                [Context Assembler]
           (Reconstructs complete blocks, orders hierarchically)
                        │
             [Ollama LLM (llama3.2:3b)]
           (Strict prompt: No hallucinations, preserve identifiers)
                        │
                  [Final Answer]
           + Exact Source Citations (Document, Page, Section)
           + Live Retrieval Inspector (Debug Breakdown)
```

### Why are these components needed?
1. **Multi-Format Parser**: Technical manuals come in PDF, DOCX, and Markdown formats with embedded code files. This layer extracts accurate page numbers and section headings for precise citations.
2. **Code & Completeness Detector**: Naive splitters don't know the difference between a 1-line call like `$(container).dxDataGrid(gridConfig);` and a 30-line complete setup with `dataSource`, `columns`, and event handlers. Our detector scores completeness (0.0 to 1.0) using structural signals.
3. **Boundary-Preserving Chunker**: Never slices across a function or configuration block midway. If an implementation is large, it generates parent-child chunks linked by `implementation_id` for perfect reconstruction.
4. **Query Intent Classifier**: Distinguishes between *"Give me complete code for X"* (`COMPLETE_CODE`), *"What is X?"* (`EXPLANATION`), and *"How do I configure X?"* (`CONFIGURATION`).
5. **Hybrid Retrieval (Vector + BM25)**: Exact identifier names like `dxDataGrid`, `gridConfig`, and `MaterialInventoryAct` can be blurred by pure semantic vector search. BM25 ensures exact keyword matches get maximum weight.
6. **Code-Aware Reranker**: For `COMPLETE_CODE` queries, it boosts chunks with `chunk_type == "complete_code"` and high completeness scores, while penalizing tiny snippets.
7. **Anti-Hallucination Prompting**: Instructs `llama3.2:3b` to answer *strictly* from the retrieved documentation context and explicitly state if only a snippet is documented.

---

## 2. Prerequisites (Windows)

Ensure you have installed:
1. **Python 3.10+**: Available from [python.org](https://www.python.org/) (check "Add Python to PATH" during install).
2. **Ollama**: Download and install from [ollama.com](https://ollama.com/).
3. **PowerShell**: Built into Windows 10/11.

---

## 3. Step-by-Step Setup Guide (Windows PowerShell)

Open **PowerShell** and navigate to your project directory:

```powershell
cd E:\Technical_Doc_RAG
```

### Step 1: Create Virtual Environment
```powershell
python -m venv venv
```

Activate the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell shows an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then retry).*

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Verify Ollama Status
Make sure the Ollama application is running (you should see the Ollama llama icon in your Windows system tray), or run:
```powershell
ollama --version
```

### Step 4: Verify `llama3.2:3b` Model
Check that `llama3.2:3b` is downloaded:
```powershell
ollama list
```
If `llama3.2:3b` is not listed, download it:
```powershell
ollama pull llama3.2:3b
```

### Step 5: Pull the Embedding Model
Pull the recommended technical embedding model:
```powershell
ollama pull nomic-embed-text
```
*(Note: If you skip this, the application automatically uses its built-in fast local ONNX embedding fallback, so your workflow is never interrupted).*

### Step 6: Start the Application
Run the startup launcher:
```powershell
python run.py
```
You will see the startup banner verifying Ollama connectivity and model availability, followed by the Uvicorn server starting on port 8000.

### Step 7: Open the Web UI
Open your web browser and navigate to:
```text
http://localhost:8000
```

---

## 4. How to Use the Application

### 1. Uploading Documentation
- In the left sidebar dropzone, drag and drop any technical manual (e.g. `data/documents/AMS_Specification_Sample.md`, or your own `.pdf`, `.docx`, `.txt`, `.md`, or code file).
- The ingestion progress bar will display while the parser extracts structure, scores code completeness, and indexes the document into ChromaDB.
- Your document will appear in the **Indexed Docs** list showing chunk counts and page counts.

### 2. Asking Questions
Try the three sample questions:
1. **Complete Implementation Query**:
   > *"Give me complete code for creating a Segment grid"*
   - The query classifier labels this as `[COMPLETE_CODE]`.
   - The reranker boosts Page 17's **complete implementation** (with `gridConfig`, `dataSource`, columns, and initialization) to rank **#1**, ignoring the isolated Page 16 snippet!
2. **Conceptual Explanation Query**:
   > *"What is a Segment?"*
   - The query classifier labels this as `[EXPLANATION]`.
   - The system retrieves Page 10's prose definition.
3. **Configuration Query**:
   > *"How do I configure Segment grid columns and dataSource?"*
   - The query classifier labels this as `[CONFIGURATION]`.
   - Returns parameters and options with exact citations.

### 3. Debug Retrieval Mode (Inspector)
- Toggle the **Debug Retrieval** switch in the sidebar or click **Inspector** in the top-right header.
- The inspector drawer displays:
  - **Query Classification**: Intent detected by regex and semantics.
  - **Expanded Queries**: Internal query variations used to cover synonyms.
  - **Extracted Identifiers**: Exact tokens like `dxDataGrid`, `gridConfig`.
  - **Candidate Breakdown**: Detailed table showing Semantic score, BM25 Keyword score, Completeness score, Section score, and Final combined score.

---

## 5. API Endpoints

The FastAPI backend provides clean REST endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check verifying Ollama, active models, and chunk counts |
| `POST` | `/api/documents/upload` | Upload & index PDF, DOCX, TXT, MD, or code files |
| `GET` | `/api/documents` | List all indexed documents with chunk statistics |
| `POST` | `/api/documents/{id}/index` | Re-index an existing document from disk |
| `DELETE` | `/api/documents/{id}` | Delete a document and its vectors from ChromaDB |
| `DELETE` | `/api/documents` | Clear all documents and reset the database |
| `POST` | `/api/search` | Debug retrieval endpoint: scores candidates without calling LLM |
| `POST` | `/api/chat` | Main RAG chat endpoint with streaming SSE support |

---

## 6. Running Automated Tests

Run the full automated test suite using `pytest`:

```powershell
pytest -v
```

Tests include:
- `tests/test_chunking.py`: Language detection, completeness scoring (asserts complete code > 0.80, snippet < 0.35).
- `tests/test_retrieval.py`: Query classification, identifier extraction, query expansion.
- `tests/test_code_queries.py`: Tests that *"Give me complete code for creating a Segment grid"* reliably ranks the Page 17 complete implementation ahead of the Page 16 snippet.

---

## 7. Configuration Options (`.env`)

All parameters are configurable via `.env`:

```env
# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2:3b

# Embedding settings
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text

# Hybrid Retrieval weights
WEIGHT_SEMANTIC=0.35
WEIGHT_KEYWORD=0.25
WEIGHT_COMPLETENESS=0.30
WEIGHT_SECTION=0.10
COMPLETE_CODE_BOOST=0.40

# Budgets
MAX_CONTEXT_TOKENS=2500
TOP_K_RETRIEVAL=15
TOP_K_RERANKED=5
```
