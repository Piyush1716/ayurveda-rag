from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.rag import ask_ayurveda
from app.models import QueryRequest

# uvicorn main:app --reload
app = FastAPI(
    title="Ayurveda RAG API",
    description="Ask questions from Charaka Samhita",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev (later restrict this)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Ayurveda RAG API is running"}

@app.get("/home")
def home():
    return FileResponse("index.html")

@app.post("/ask")
def ask_question(request : QueryRequest):
    result = ask_ayurveda(request.question)
    return result
