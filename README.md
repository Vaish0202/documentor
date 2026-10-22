# DocuMentor

Study assistant over your own documents: grounded Q&A and quiz generation.
Built with LangChain (LCEL), LangServe and LangSmith.

## Run locally (Ollama)
```bash
ollama pull llama3.2
pip install -r requirements.txt
cp .env.example .env          # fill LANGSMITH_API_KEY
# put PDFs / .txt / .md files in ./data
python ingest.py
uvicorn server:app --reload --port 8000
```
Open http://localhost:8000/qa/playground/

## Deploy
Set `LLM_PROVIDER=groq`, `GROQ_API_KEY`, and a model name that exists in
`GET https://api.groq.com/openai/v1/models`. The model name is an env var, so
swapping it never needs a code change.
