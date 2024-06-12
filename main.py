import os

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from typing import Dict
from langchain_pinecone import PineconeVectorStore

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()


class Query(BaseModel):
    question: str


app = FastAPI()

# Get the project's root directory
project_root = os.getcwd()

uploaded_document_path = None
uploaded_document_id = None
doc_search = None


def initialize_doc_search(file_path: str):
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file format")

        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        docs = splitter.split_documents(pages)
        embeddings = HuggingFaceEmbeddings()
        index_name = "trial"
        doc_search = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
        print(doc_search)
        # vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        # doc_search = vectorstore.from_documents(docs, embeddings)
        return doc_search

    except Exception as e:
        print(f"Error initializing document search: {e}")
        return None


@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    global uploaded_document_path, uploaded_document_id, doc_search

    try:
        file_path = os.path.join(project_root, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        uploaded_document_path = file_path
        uploaded_document_id = file.filename
        doc_search = initialize_doc_search(file_path)
        print(doc_search)

        return {"document_id": uploaded_document_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/qa/")
async def query_document(query: Query):
    if not doc_search:
        raise HTTPException(status_code=400, detail="Document search not initialized")

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=100,
        do_sample=False,
    )

    retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=doc_search.as_retriever())
    result = await retrieval_chain.acall(query.question)

    return JSONResponse(status_code=200, content={"answer": result["result"]})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
