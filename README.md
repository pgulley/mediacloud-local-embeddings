Featherweight Embeddings
========================

Small experiment using EmbeddingGemma for creating a semantically searchable local index

## Run

### Gradio Web Interface (Recommended)
```bash
python gradio_app.py
```
Then open http://localhost:7860 in your browser.

### Streamlit Interface (Legacy)
```bash
streamlit run streamlit_app.py
```

## Required Settings
- MC_API_KEY - full-text access token for mediacloud
- HF_TOKEN - 'read' token from huggingface for downloading model

### Optional LLM Settings (env vars, used by summarization)
These use the `LLM_` prefix and are read by `LLMSettings`.
- LLM_PROVIDER - LLM provider id (default: `ollama`)
- LLM_MODEL - model name or tag (default: `llama3:8b`)
- LLM_TEMPERATURE - float, 0.0–1.0 (default: `0.2`)
- LLM_MAX_OUTPUT_TOKENS - integer max tokens for summary (default: `512`)
- LLM_OLLAMA_HOST - Ollama server URL (default: `http://localhost:11434`)

Example `.env` snippet:
```
MC_API_KEY=...
HF_TOKEN=...

LLM_PROVIDER=ollama
LLM_MODEL=llama3:8b
LLM_TEMPERATURE=0.2
LLM_MAX_OUTPUT_TOKENS=512
LLM_OLLAMA_HOST=http://localhost:11434
```

## Features

### Core Functionality
- **Semantic Search**: Build searchable indexes from MediaCloud queries using EmbeddingGemma
- **AI Summarization**: Generate summaries with citations using local LLMs (via Ollama)
- **Question Answering**: Ask questions about your indexed content
- **Attention Analytics**: Track story attention over time

### 🆕 Clustering Analysis (New!)
- **K-means Clustering**: Automatically discover thematic clusters in your content
- **UMAP Visualization**: 2D scatter plots showing content clusters
- **Cluster Analysis**: Detailed breakdowns of discovered themes and topics
- **Interactive Plots**: Hover over points to see story titles and content
- **Direct FAISS Integration**: Extracts embeddings directly from FAISS index for maximum efficiency

### Using Clustering Analysis
1. Build an index from MediaCloud in the "Build Index" tab
2. Go to the "Clustering" tab
3. Choose auto-detect clusters or set a specific number
4. Click "Run Clustering Analysis" to see clustering and visualization
5. Explore the interactive UMAP plot to understand content patterns

## Demo Scripts
- `demo_clustering.py` - Standalone demo of clustering analysis features

## Plans

### Building the Semantic Index
- Term Expansion on query before sending to MC
- Index on disk? Currently just store the disk in memory using FAISS.
- Rerank search results, better thresholding
- Support for citations in search results- pass-through story ids and links. 
- MCP-ify- turn the index into a tool that an agent could interface with


