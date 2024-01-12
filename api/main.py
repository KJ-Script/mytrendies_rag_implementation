# os and langchain imports
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# fast api imports
from typing import Union
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

# initalizing fastapi, setting up cors origins and middle
app = FastAPI()

origins = [
    "*"
]

app.add_middleware(CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# initalizing key and embeddings
os.environ["OPENAI_API_KEY"] = "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"
directory = 'database'
embedding = OpenAIEmbeddings()


# initalizing vector databases and retrievers
vectordb=Chroma(persist_directory=directory, embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)

# llm functions
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
    return llm_response['result']



class Item(BaseModel):
    data: str

@app.get("/")
def read_root():
    return {"Hello": "there"}


@app.post("/sendData")
async def processData(request: Request):
    print("Item", request.body)
    # llm_response = qa_chain(item.data)
    # response = process_llm_response(llm_response)
    # return response
    return request.body

@app.get("/testing")
def test():
    return "hey"

@app.post("/createResponse/")
async def processData(item: Item):
    print("Item", item.data)
    llm_response = qa_chain(item.data)
    response = process_llm_response(llm_response)
    print(response)
    return response