# Lab RAG System

A comprehensive RAG (Retrieval-Augmented Generation) system that combines PDF document search, SQL database queries, and LLM-powered AI agents with a web interface.

## Overview

This project provides:
- **PDF Document RAG**: Search and retrieve information from research papers stored in Milvus vector database
- **Database Content Search**: Semantic search over embedded MSSQL textual data
- **AI Agent Framework**: LangChain-powered agent that can use multiple tools (PDF search, vector search, calculations)
- **Web Interface**: Open WebUI for easy interaction with the AI system
- **Vector Search**: Semantic search capabilities for both documents and database content

## Architecture

- **Milvus**: Vector database for storing PDF embeddings and MSSQL data embeddings
- **Ollama**: Local LLM service for text generation and embeddings
- **FastAPI**: REST API backend implementing OpenAI-compatible endpoints
- **LangChain**: Agent framework for tool orchestration and reasoning
- **Open WebUI**: Web interface for chat interactions

## Services

| Service | Container | Ports | Description |
|---------|-----------|-------|-------------|
| API | lab-rag-api-alan | 8081:1010 | Main RAG API service |
| Open WebUI | openwebui-alan | 3001:8080 | Web chat interface |
| Ollama | ollama-alan | 11434:11434 | Local LLM service |
| Milvus | milvus-standalone-alan | 19531:19530 | Vector database |
| MinIO | milvus-minio-alan | 9002:9000, 9003:9001 | Object storage for Milvus |
| etcd | milvus-etcd-alan | - | Metadata storage for Milvus |

## Quick Start

### Prerequisites
- Docker and Docker Compose
- NVIDIA Docker runtime (for GPU support)
- GPU with CUDA support

### Setup
1. Clone the repository
2. Copy and configure environment file:
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```
3. Place PDF files in `data/pdf/` directory
4. Start the services:
   ```bash
   docker compose up -d
   ```

### Access Points
- **Web Interface**: http://localhost:3001
- **API Endpoint**: http://localhost:8081
- **Milvus**: localhost:19531
- **MinIO Console**: http://localhost:9003

## API Usage

### OpenAI-Compatible Endpoints

#### List Models
```bash
curl http://localhost:8081/v1/models
```

#### Chat Completions
```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "What is Northwind database?"}],
    "stream": false
  }'
```

#### Streaming Chat
```bash
curl -N -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name", 
    "messages": [{"role": "user", "content": "Explain machine learning"}],
    "stream": true
  }'
```

## Agent Tools

The AI agent has access to several tools:

1. **LabPaperSearch**: RAG search over PDF documents for research questions
2. **MSSQLVectorSearch**: Semantic search over embedded database content
3. **Python_REPL**: Execute Python code for calculations and data analysis
4. **LLMAnswer**: General purpose text generation and explanations

## Configuration

Key environment variables in `.env`:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=your-model-name
OLLAMA_EMBED_MODEL=your-embedding-model

# Milvus Configuration  
MILVUS_HOST=standalone
MILVUS_PORT=19530
COLLECTION_NAME_1=collection-name

# Database Configuration (Optional - for vector embeddings only)
MSSQL_SERVER=your-sql-server
MSSQL_DATABASE=your-database
MSSQL_USER=your-username
MSSQL_PASSWORD=your-password
```

## Development

### Data Ingestion
```bash
# Ingest PDF files into Milvus
python scripts/ingest_pdfs.py

# Ingest SQL data into Milvus
python scripts/ingest_sql.py
```

### Testing
```bash
# Run basic tests
python test_agent.py
python test_lab.py
```

## Docker Management Commands

### Complete Reset
```bash
docker ps -aq | xargs -r docker rm -f
docker images -aq | xargs -r docker rmi -f
docker volume ls -q | xargs -r docker volume rm
docker system prune --volumes -f
```

### Service Management
```bash
# Rebuild specific service
docker compose build api
docker compose up -d api

# Reset Open WebUI
docker compose down openwebui
docker compose up -d openwebui

# View logs
docker logs lab-rag-api-alan
docker logs openwebui-alan
```

### Container Status
```bash
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}"
```

## Troubleshooting

### Common Issues

1. **Models not showing in Open WebUI**:
   - Check API endpoint configuration: `OPENAI_API_BASE=http://lab-rag-api-alan:1010`
   - Verify `/v1/models` endpoint returns valid JSON
   - Clear browser cache and reload

2. **Agent parsing errors**:
   - Check LLM output format
   - Verify tool descriptions and system messages
   - Review agent iteration limits

3. **PDF search not working**:
   - Ensure PDFs are ingested: `python scripts/ingest_pdfs.py`
   - Check Milvus collection exists
   - Verify embedding model is loaded

4. **Vector search not working**:
   - Check Milvus connection settings
   - Verify data collections exist in Milvus
   - Review embedding model configuration

### Logs and Debugging
```bash
# View all service logs
docker compose logs

# Follow specific service logs
docker logs -f lab-rag-api-alan

# Check container health
docker ps
```

## License

This project is licensed under the MIT License.
