// ==========================================================================
// Technical Doc RAG — Frontend Client Logic
// ==========================================================================

const API_BASE = window.location.origin + "/api";

// DOM Elements
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const statusModel = document.getElementById("status-model");

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const uploadProgress = document.getElementById("upload-progress-container");
const uploadProgressText = document.getElementById("upload-progress-text");

const documentList = document.getElementById("document-list");
const docCountSpan = document.getElementById("doc-count");
const clearAllBtn = document.getElementById("clear-all-btn");

const chatHistory = document.getElementById("chat-history");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

const debugToggle = document.getElementById("debug-toggle");
const debugDrawer = document.getElementById("debug-drawer");
const openDebugBtn = document.getElementById("open-debug-btn");
const closeDebugBtn = document.getElementById("close-debug-btn");
const debugContent = document.getElementById("debug-content");
const suggestionsContainer = document.getElementById("suggestions-container");

// State
let isStreaming = false;

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    fetchDocuments();
    setupEventListeners();
    setupTextareaAutoResize();
});

function setupEventListeners() {
    // Dropzone upload
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--accent-primary)";
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "var(--border-highlight)";
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--border-highlight)";
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    // Clear all documents
    clearAllBtn.addEventListener("click", async () => {
        if (confirm("Are you sure you want to clear all indexed documents and vectors?")) {
            try {
                await fetch(`${API_BASE}/documents`, { method: "DELETE" });
                fetchDocuments();
            } catch (err) {
                alert("Failed to clear documents: " + err);
            }
        }
    });

    // Chat submit
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        submitQuery();
    });

    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submitQuery();
        }
    });

    // Debug drawer toggle
    openDebugBtn.addEventListener("click", () => {
        debugDrawer.classList.toggle("closed");
    });

    closeDebugBtn.addEventListener("click", () => {
        debugDrawer.classList.add("closed");
    });

    // Suggestion chips
    document.querySelectorAll(".suggestion-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            userInput.value = chip.dataset.query;
            submitQuery();
        });
    });
}

function setupTextareaAutoResize() {
    userInput.addEventListener("input", () => {
        userInput.style.height = "auto";
        userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
    });
}

// ==========================================================================
// Health & Ollama Status Check
// ==========================================================================
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();

        if (data.ollama && data.ollama.status === "online") {
            statusDot.className = "status-indicator-dot online";
            statusText.textContent = "Ollama Connected";
            statusModel.textContent = `${data.llm_model} • ${data.total_chunks_count} chunks`;
        } else {
            statusDot.className = "status-indicator-dot offline";
            statusText.textContent = "Ollama Offline";
            statusModel.textContent = "Start Ollama on port 11434";
        }
    } catch (e) {
        statusDot.className = "status-indicator-dot offline";
        statusText.textContent = "Backend Offline";
        statusModel.textContent = "Check server console";
    }
}

// ==========================================================================
// Document Management
// ==========================================================================
async function fetchDocuments() {
    try {
        const res = await fetch(`${API_BASE}/documents`);
        const docs = await res.json();
        renderDocuments(docs);
        checkHealth();
    } catch (err) {
        console.error("Failed to load documents", err);
    }
}

function renderDocuments(docs) {
    docCountSpan.textContent = docs.length;
    documentList.innerHTML = "";

    if (docs.length === 0) {
        documentList.innerHTML = `
            <div class="empty-docs">
                No documents indexed yet. Upload a technical document to begin.
            </div>
        `;
        return;
    }

    docs.forEach(doc => {
        const card = document.createElement("div");
        card.className = "doc-card";

        const typeBadgeClass = `badge-${doc.file_type || "txt"}`;

        card.innerHTML = `
            <div class="doc-card-top">
                <span class="doc-title" title="${doc.filename}">${doc.filename}</span>
                <div class="doc-actions">
                    <button class="doc-action-btn reindex-btn" title="Re-index document" data-id="${doc.doc_id}">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="23 4 23 10 17 10"></polyline>
                            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                        </svg>
                    </button>
                    <button class="doc-action-btn delete-btn" title="Delete document" data-id="${doc.doc_id}">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    </button>
                </div>
            </div>
            <div class="doc-badges">
                <span class="badge ${typeBadgeClass}">${doc.file_type}</span>
                <span class="badge badge-chunks">${doc.chunk_count} chunks</span>
                ${doc.page_count > 1 ? `<span class="badge badge-chunks">${doc.page_count} pgs</span>` : ""}
            </div>
        `;

        // Bind delete
        card.querySelector(".delete-btn").addEventListener("click", async (e) => {
            e.stopPropagation();
            if (confirm(`Delete '${doc.filename}'?`)) {
                await fetch(`${API_BASE}/documents/${doc.doc_id}`, { method: "DELETE" });
                fetchDocuments();
            }
        });

        // Bind re-index
        card.querySelector(".reindex-btn").addEventListener("click", async (e) => {
            e.stopPropagation();
            uploadProgress.style.display = "block";
            uploadProgressText.textContent = `Re-indexing ${doc.filename}...`;
            try {
                await fetch(`${API_BASE}/documents/${doc.doc_id}/index`, { method: "POST" });
                fetchDocuments();
            } finally {
                uploadProgress.style.display = "none";
            }
        });

        documentList.appendChild(card);
    });
}

async function handleFileUpload(file) {
    uploadProgress.style.display = "block";
    uploadProgressText.textContent = `Indexing ${file.name}...`;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API_BASE}/documents/upload`, {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            alert(`Error: ${err.detail || "Failed to upload document"}`);
        } else {
            const result = await res.json();
            fetchDocuments();
        }
    } catch (e) {
        alert("Upload error: " + e.message);
    } finally {
        uploadProgress.style.display = "none";
        fileInput.value = "";
    }
}

// ==========================================================================
// Chat & Streaming Interaction
// ==========================================================================
async function submitQuery() {
    const text = userInput.value.trim();
    if (!text || isStreaming) return;

    userInput.value = "";
    userInput.style.height = "auto";
    isStreaming = true;
    sendBtn.disabled = true;

    // Append user message
    appendUserMessage(text);

    // Create assistant message container
    const { bodyElem, metaElem, contentElem, citationContainer } = createAssistantMessage();

    const isDebug = debugToggle.checked;

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: text,
                stream: true,
                debug: isDebug
            })
        });

        if (!response.ok) {
            contentElem.textContent = `Server Error (${response.status}): ${await response.text()}`;
            isStreaming = false;
            sendBtn.disabled = false;
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let accumulatedTokens = "";
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            buffer = lines.pop(); // keep last incomplete chunk

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                const dataStr = line.replace("data: ", "").trim();
                if (!dataStr) continue;

                try {
                    const parsed = JSON.parse(dataStr);

                    if (parsed.event === "metadata") {
                        // Render query class badge
                        renderQueryBadge(metaElem, parsed.query_class);

                        // Render citations
                        if (parsed.sources && parsed.sources.length > 0) {
                            renderCitations(citationContainer, parsed.sources);
                        }

                        // Populate debug drawer if debug data returned
                        if (parsed.debug_info) {
                            renderDebugData(parsed.debug_info);
                        }
                    } else if (parsed.event === "token") {
                        accumulatedTokens += parsed.token;
                        contentElem.innerHTML = marked.parse(accumulatedTokens);
                        highlightCodeBlocks(contentElem);
                        chatHistory.scrollTop = chatHistory.scrollHeight;
                    } else if (parsed.event === "done") {
                        // Final highlight pass
                        highlightCodeBlocks(contentElem);
                    }
                } catch (e) {
                    console.error("Error parsing SSE line", e, dataStr);
                }
            }
        }
    } catch (err) {
        contentElem.innerHTML = `<p style="color: var(--accent-rose);">Error connecting to RAG backend: ${err.message}</p>`;
    } finally {
        isStreaming = false;
        sendBtn.disabled = false;
    }
}

function appendUserMessage(text) {
    const msg = document.createElement("div");
    msg.className = "message user-message";
    msg.innerHTML = `
        <div class="message-avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
            </svg>
        </div>
        <div class="message-body">
            <div class="message-meta">
                <span class="sender-name">You</span>
            </div>
            <div class="message-content">
                ${escapeHtml(text)}
            </div>
        </div>
    `;
    chatHistory.appendChild(msg);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function createAssistantMessage() {
    const msg = document.createElement("div");
    msg.className = "message assistant-message";
    msg.innerHTML = `
        <div class="message-avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                <line x1="8" y1="21" x2="16" y2="21"></line>
                <line x1="12" y1="17" x2="12" y2="21"></line>
                <polyline points="7 8 10 10.5 7 13"></polyline>
                <line x1="13" y1="13" x2="17" y2="13"></line>
            </svg>
        </div>
        <div class="message-body">
            <div class="message-meta">
                <span class="sender-name">Clue</span>
                <span class="query-badge-container"></span>
            </div>
            <div class="message-content markdown-body">
                <span class="generating-indicator">Retrieving & generating...</span>
            </div>
            <div class="citation-container" style="display: none;">
                <span class="citation-header">Sources Cited:</span>
                <div class="citation-list"></div>
            </div>
        </div>
    `;
    chatHistory.appendChild(msg);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    return {
        bodyElem: msg.querySelector(".message-body"),
        metaElem: msg.querySelector(".query-badge-container"),
        contentElem: msg.querySelector(".message-content"),
        citationContainer: msg.querySelector(".citation-container")
    };
}

function renderQueryBadge(container, queryClass) {
    const badgeClassMap = {
        "COMPLETE_CODE": "badge-complete-code",
        "CODE": "badge-code",
        "EXPLANATION": "badge-explanation",
        "CONFIGURATION": "badge-configuration",
        "TROUBLESHOOTING": "badge-troubleshooting"
    };

    const cls = badgeClassMap[queryClass] || "badge-info";
    container.innerHTML = `<span class="query-badge ${cls}">${queryClass}</span>`;
}

function renderCitations(container, sources) {
    container.style.display = "flex";
    const list = container.querySelector(".citation-list");
    list.innerHTML = "";

    sources.forEach(src => {
        const pill = document.createElement("div");
        pill.className = "citation-pill";
        pill.title = src.snippet;
        pill.innerHTML = `
            <span>📄 ${src.source}</span>
            <span>p. ${src.page}</span>
            <span>• ${src.section}</span>
            <span style="opacity: 0.7; font-size: 0.65rem;">[${src.chunk_type}]</span>
        `;
        list.appendChild(pill);
    });
}

function highlightCodeBlocks(container) {
    container.querySelectorAll("pre code").forEach(block => {
        Prism.highlightElement(block);

        // Add copy button if not present
        const pre = block.parentElement;
        if (!pre.querySelector(".copy-code-btn")) {
            const copyBtn = document.createElement("button");
            copyBtn.className = "copy-code-btn";
            copyBtn.textContent = "Copy";
            copyBtn.addEventListener("click", () => {
                navigator.clipboard.writeText(block.textContent);
                copyBtn.textContent = "Copied!";
                setTimeout(() => { copyBtn.textContent = "Copy"; }, 2000);
            });
            pre.appendChild(copyBtn);
        }
    });
}

// ==========================================================================
// Debug Drawer / Retrieval Inspector
// ==========================================================================
function renderDebugData(debugInfo) {
    debugContent.innerHTML = `
        <div class="debug-card">
            <span class="debug-card-title">Query Classification</span>
            <div><span class="query-badge badge-complete-code" style="font-size: 0.8rem;">${debugInfo.query_class}</span></div>
        </div>

        <div class="debug-card">
            <span class="debug-card-title">Expanded Search Queries</span>
            <div class="debug-pill-list">
                ${debugInfo.expanded_queries.map(q => `<span class="debug-pill">${escapeHtml(q)}</span>`).join("")}
            </div>
        </div>

        ${debugInfo.extracted_identifiers.length > 0 ? `
            <div class="debug-card">
                <span class="debug-card-title">Technical Identifiers</span>
                <div class="debug-pill-list">
                    ${debugInfo.extracted_identifiers.map(id => `<span class="debug-pill" style="color: var(--accent-emerald);">${escapeHtml(id)}</span>`).join("")}
                </div>
            </div>
        ` : ""}

        <div class="debug-card">
            <span class="debug-card-title">Retrieved & Reranked Candidates</span>
            <div class="candidates-list">
                ${debugInfo.results.map((c, i) => `
                    <div class="candidate-item">
                        <div class="candidate-header">
                            <span class="candidate-rank">#${i + 1} ${c.chunk_type.toUpperCase()}</span>
                            <span class="candidate-score">Final: ${c.final_score.toFixed(3)}</span>
                        </div>
                        <div style="font-size: 0.72rem; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                            ${c.source} • Page ${c.page} • ${c.section}
                        </div>
                        <div class="score-bars">
                            <div class="score-row"><span>Semantic:</span><span>${c.semantic_score.toFixed(3)}</span></div>
                            <div class="score-row"><span>Keyword (BM25):</span><span>${c.keyword_score.toFixed(3)}</span></div>
                            <div class="score-row"><span>Completeness:</span><span>${c.completeness_score.toFixed(2)}</span></div>
                            <div class="score-row"><span>Section Boost:</span><span>${c.section_score.toFixed(2)}</span></div>
                        </div>
                    </div>
                `).join("")}
            </div>
        </div>
    `;
}

function escapeHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
