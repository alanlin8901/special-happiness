#!/bin/bash
# if ! python3 -c "import llama_cpp" &>/dev/null; then
#     echo "Installing llama-cpp-python with CUDA support..."
#     export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75"
#     pip install --no-cache-dir "git+https://github.com/abetlen/llama-cpp-python.git@0b89fe48ad26ffcff76451bd87642d916a1a3385"
# fi

echo "🕐 Waiting for Milvus to be ready..."
python scripts/ingest_pdfs.py || echo "⚠️ PDF ingestion failed or was already done."
python scripts/ingest_sql.py || echo "⚠️ SQL ingestion failed or was already done."
echo "🚀 Starting uvicorn..."

exec "$@"
