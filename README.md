# financial-assistant
Agentic Financial Research Assistant
An end-to-end AI/ML portfolio project: a tool-calling agent that answers corporate-banking
questions from policy documents using retrieval-augmented generation (RAG), plus a
transaction anomaly-detection model, served through a FastAPI REST API.
It mirrors the kind of system used in production banking AI: document retrieval for analyst
research, agent tool use for multi-step questions, and ML-based risk monitoring — built here
entirely on sample data so it runs anywhere with no API keys.
What it does
RAG document search — indexes policy documents (lending, fraud monitoring, account
  opening, wire transfers), chunks them, and retrieves the most relevant passages for a query.
Tool-calling agent — routes each question to tools (search_documents,
  check_anomalies), collects evidence, and composes an answer with sources cited.
Anomaly detection — an Isolation Forest model trained on synthetic transaction data
  flags unusual activity (large amounts, off-hours, high frequency). Training is tracked
  with MLflow.
Optional LLM answers — set OPENAI_API_KEY and the agent writes its final answer with
  an LLM grounded in the retrieved context; without a key, an extractive fallback answers
  from the evidence so the demo always works.
Quickstart
pip install -r requirements.txt
uvicorn app.main:app --reload
Then open http://localhost:8000/docs for the interactive API explorer.
Example requests
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What approval is required for a loan above $1,000,000?"}'

curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Were there any unusual or suspicious transactions?"}'

curl "http://localhost:8000/search?q=wire+transfer+cutoff"
curl http://localhost:8000/anomalies
Train the anomaly model (with MLflow tracking)
python train_model.py
mlflow ui   # view the experiment at http://localhost:5000
Run the tests
pytest
Run with Docker
docker build -t agentic-financial-assistant .
docker run -p 8000:8000 agentic-financial-assistant
Project structure
app/
  main.py      FastAPI service (/health, /ask, /search, /anomalies)
  agent.py     Tool-calling agent (routing, tool use, answer composition)
  rag.py       Document chunking + retrieval (TF-IDF/cosine; swappable for dense embeddings)
  anomaly.py   Synthetic transaction data + Isolation Forest detector
data/sample_docs/   Sample banking policy documents
tests/         Unit + API tests (pytest)
train_model.py MLflow-tracked training entry point
Production notes / roadmap
Swap TF-IDF retrieval for dense embeddings (sentence-transformers or OpenAI embeddings)
  with a vector database such as FAISS, Pinecone, or pgvector — the DocumentStore
  interface stays the same.
Replace the synthetic transactions with real account activity and add supervised models
  (XGBoost/LightGBM) alongside the unsupervised detector.
Add an evaluation harness for answer groundedness and retrieval quality, authentication,
  and structured logging/monitoring for the API.
Skills demonstrated
Python · FastAPI · REST APIs · LLMs · Agentic AI · Tool Calling · RAG · Semantic Search ·
Scikit-learn · Isolation Forest · MLflow · Pandas · NumPy · Docker · PyTest · CI-ready layout
