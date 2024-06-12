import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader  # document loader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # document transformer: text splitter for chunking
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma  # vector store
from langchain import HuggingFaceHub  # model hub
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import Docx2txtLoader
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()


class Query(BaseModel):
    question: str


app = FastAPI()

doc_search = None  # Initialize the global doc_search variable


def initialize_doc_search(file_path: str):
    try:
        # Load the document based on its file extension
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file format")

        # Process the document
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        docs = splitter.split_documents(pages)
        embeddings = HuggingFaceEmbeddings()

        global doc_search
        doc_search = Chroma.from_documents(docs, embeddings)

    except Exception as e:
        print(f"Error initializing document search: {e}")


# Define the path to the document
document_path = os.environ.get("DOCUMENT_PATH")  # Change this to your document's path

# Initialize the document search on startup
initialize_doc_search(document_path)


@app.post("/query")
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

# import os
# import getpass
# from dotenv import load_dotenv
#
# from langchain.document_loaders import PyPDFLoader  # document loader: https://python.langchain.com/docs/modules/data_connection/document_loaders
# from langchain.text_splitter import RecursiveCharacterTextSplitter  # document transformer: text splitter for chunking
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma  # vector store
# from langchain import HuggingFaceHub  # model hub
# from langchain.chains import RetrievalQA
# import chainlit as cl
# from langchain_community.document_loaders import Docx2txtLoader
# from langchain_huggingface import HuggingFaceEndpoint
#
# load_dotenv()
#
# # path = input("Enter PDF file path: ")
# # loader = PyPDFLoader(path)
# # pages = loader.load()
# # print(len(pages))
#
# path = input("Enter Word file path: ")
# loader = Docx2txtLoader(path)
# pages = loader.load()
# print(len(pages))
#
# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
# docs = splitter.split_documents(pages)
# print(len(docs))
# embeddings = HuggingFaceEmbeddings()
# doc_search = Chroma.from_documents(docs, embeddings)
#
# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#     task="text-generation",
#     max_new_tokens=100,
#     do_sample=False,
# )
#
#
# @cl.on_chat_start
# def main():
#     retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=doc_search.as_retriever())
#     cl.user_session.set("retrieval_chain", retrieval_chain)
#
# # Function to handle incoming messages
# @cl.on_message
# async def main(message: str):
#     retrieval_chain = cl.user_session.get("retrieval_chain")
#     res = await retrieval_chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
#     await cl.Message(content=res["result"]).send()


# from groq import Groq
# from dotenv import load_dotenv
# import os
#
# load_dotenv()
# groqcloud_api_key = os.getenv("GROQCLOUD_API_KEY")
# client = Groq(
#     api_key=os.environ.get("GROQCLOUD_API_KEY"),
# )
# prompt = input("Enter your prompt: ")
# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "system",
#             "content": "you are a helpful assistant."
#         },
#         {
#             "role": "user",
#             "content": prompt,
#         }
#     ],
#     model="llama3-8b-8192",
# )
# print(chat_completion.choices[0].message.content)
