Lightweight Embeddings Demo
===========================

Small experiment using EmbeddingGemma for creating a semantically searchable local index

## Required Settings
- MC_API_KEY - full-text access token for mediacloud
- HF_TOKEN - 'read' token from huggingface for downloading model

## Plans

### Building the Semantic Index
- Term Expansion on query before sending to MC
- Index on disk? Currently just store the disk in memory using FAISS.
- Rerank search results, better thresholding 
- Support for citations in search results- pass-through story ids and links. 

- MCP-ify- turn the index into a tool that an agent could interface with

### Demo UI/UX


