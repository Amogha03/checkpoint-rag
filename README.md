# Checkpoint RAG

This project demonstrates a retrieval-augmented generation (RAG) workflow using LangChain, OpenAI, and Pinecone.


## Prerequisites

- Python 3.10 or newer
- An OpenAI API key
- A Pinecone API key
- A Pinecone index available in your account

## 1) Clone and enter the repository

```bash
git clone <your-repository-url>
cd checkpoint-rag
```

## 2) Create a virtual environment

```bash
# If needed, install Python 3.10 first
# brew install python@3.10

python3.10 -m venv .venv
source .venv/bin/activate
```

If `python3.10` is not available, install it first and then rerun the commands above.

## 3) Install project dependencies

```bash
python -V
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m ipykernel install --user --name checkpoint-rag
```

This installs the project dependencies declared in [pyproject.toml](pyproject.toml), including `python-dotenv` and `langchain-pinecone`.

## 4) Configure environment variables

Copy the sample environment file and fill in your real credentials:

```bash
cp .env.example .env
```

Then update `.env` with your values:

```bash
OPENAI_API_KEY="your_openai_api_key_here"
PINECONE_API_KEY="your_pinecone_api_key_here"
```