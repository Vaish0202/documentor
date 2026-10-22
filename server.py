"""LangServe app.  Run:  uvicorn server:app --reload --port 8000
Playground: http://localhost:8000/qa/playground/   |  Docs: /docs
"""
from fastapi import FastAPI
from langserve import add_routes

from chains import qa_chain, quiz_chain

app = FastAPI(title="DocuMentor", version="0.1")


@app.get("/health")
def health():
    return {"status": "ok"}


add_routes(app, qa_chain, path="/qa")
add_routes(app, quiz_chain, path="/quiz")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
