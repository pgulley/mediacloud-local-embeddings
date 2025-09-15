Lightweight Embeddings Demo
===========================

Small experiment using EmbeddingGemma for creating a semantically searchable local index

## Run
`streamlit run streamlit_app.py`

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

## Plans

### Building the Semantic Index
- Term Expansion on query before sending to MC
- Index on disk? Currently just store the disk in memory using FAISS.
- Rerank search results, better thresholding 
- Support for citations in search results- pass-through story ids and links. 

- MCP-ify- turn the index into a tool that an agent could interface with

### Demo UI/UX


