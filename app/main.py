from fastapi import FastAPI, UploadFile, HTTPException
from pypdf import PdfReader
import os

from app.core.vectorstore import vectorstore
from app.routes import ask
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter


app = FastAPI()

app.include_router(ask.router)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

os.makedirs("uploads", exist_ok=True)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", ",", " ", ""]
)


def chunk_text(text):
    return text_splitter.split_text(text)


def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)

    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            full_text += page_text

    return full_text


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.post("/upload")
def upload_file(file: UploadFile):

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    contents = file.file.read()

    filepath = f"uploads/{file.filename}"

    with open(filepath, "wb") as f:
        f.write(contents)

    text = extract_text_from_pdf(filepath)
    chunks = chunk_text(text)

    ids = [
        f"{file.filename}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": file.filename,
            "chunk_index": i
        }
        for i in range(len(chunks))
    ]

    vectorstore.add_texts(
        texts=chunks,
        ids=ids,
        metadatas=metadatas
    )

    return {
        "filename": file.filename,
        "chunks": len(chunks)
    }


@app.get("/count")
def get_count():
    return {
        "total_chunks": vectorstore._collection.count()
    }