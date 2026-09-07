# SafeX AI Customer Support Chatbot



A beginner-friendly Retrieval-Augmented Generation (RAG) customer-support chatbot built with Python, LangChain, ChromaDB, local sentence-transformer embeddings, Google Gemini, and Streamlit.

## Quick start

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add a Gemini API key to `.env`, then build the vector store and start the app:

```powershell
python -m src.vector_store
streamlit run app.py
```

The `.env` file must be in the project root (beside `app.py`) and contain:

```text
GOOGLE_API_KEY=your_actual_google_gemini_api_key
```

Replace the example value with your actual key. The application resolves this
file from the project path, so the Streamlit command can be run from any
working directory. Never commit `.env`; it is excluded by `.gitignore`.

The chatbot answers only from the SafeX knowledge base. Unsupported questions receive:

> I could not find this information in the available knowledge base.

## Architecture

Documents are cleaned and loaded, split into overlapping chunks, embedded locally, and persisted in ChromaDB. Each question is searched against the vector store. Results must pass a similarity threshold before being sent to Gemini with a strict context-only prompt.

## Project structure

```text
app.py                 Streamlit interface
data/                  SafeX knowledge base
src/                   Loading, chunking, retrieval, RAG, evaluation
tests/                 Offline tests
```

## Evaluation

Run the offline tests with `pytest`. An API-backed evaluation can be run with:

```powershell
python -m src.evaluation
```

The evaluation script writes `evaluation_results.csv`; it requires `GOOGLE_API_KEY`.

## Limitations

The knowledge base is fictional and limited, retrieval quality depends on the embedding model, Gemini requires an API key and network access, and knowledge-base updates require rebuilding the vector store.
