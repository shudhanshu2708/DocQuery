from fastapi import APIRouter, HTTPException
from groq import RateLimitError
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.core.vectorstore import vectorstore
from dotenv import load_dotenv
import os

load_dotenv()

router = APIRouter(prefix="/ask", tags=["Ask"])


class AskRequest(BaseModel):
    question: str


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 30,
        "lambda_mult": 0.5
    }
)


prompt = ChatPromptTemplate.from_template("""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Use information only from the context.
2. Do not use outside knowledge.
3. If the answer is not contained in the context, say exactly:
   "I don't know based on the document."
4. If the question asks for multiple things, answer every part that is supported by the context.
5. Do not omit relevant information from the retrieved context.
6. Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
""")


@router.post("")
def ask_question(request: AskRequest):
    docs = retriever.invoke(request.question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    try:
        response = llm.invoke(
            prompt.format(
                context=context,
                question=request.question
            )
        )

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="LLM rate limit exceeded. Please try again later."
        )

    return {
        "answer": response.content,
        "sources": [doc.metadata for doc in docs]
    }