from fastapi import FastAPI

app = FastAPI(title="Filumart RAG Assistant")


@app.get("/health")
def health():
    return {"status": "ok"}
