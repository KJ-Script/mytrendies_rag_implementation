# os and langchain imports
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts.prompt import PromptTemplate

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
retriever = vectordb.as_retriever()


template = """You are a chatbot for blogging site. 
Your purpose is to respond to the questions using only the context provided
You chat, summarize, reccommend and talk about articles only from the context provided.
DO NOT USE ANYOTHER WEBSITE AS REFRENCE.
If you don't know the answer, simply state that you don't know and no more.
If the question is too general, ask the user to specify their question.

In your response, The Link/Source/Hyperlink to the article/context you refrenced MUST be included
Including the Link/Source/Hyperlink is MANDATORY

In your response, Author of the Article refrenced must also be included.

Respond in the langauge you were prompted in

Use ONLY the context Provided
If you dont know or recognize the language respond by saying you dont know or recognize the language

Providing a Link/Source/Hyperlink is Mandatory.
Remember to include Link/Source/Hyperlink section in the response
Always link to the article you were asked about or refrenced to answer the question

Use ONLY the context Provided. Do not refrence anyother article from a different source
{context}

Question: {question}"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)


# changing retriever
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                  chain_type="stuff", 
                                  retriever=retriever,
                                  chain_type_kwargs={"prompt": PROMPT, }, 
                                  return_source_documents=True)



# template = """
# Context: you are a chat bot trained on the data from a blogging site.
# when you are prompted, respond ONLY using the data from the context you are trained on. If the prompt is not related to the context reply with I dont know. Do not say anything else
# In your response the link/source/hyperlink to the article that is found in your context that you refrenced MUST be included.
# --------------------------------------
# {summaries}
# """

# qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=OpenAI(), 
#                                   chain_type="stuff", 
#                                   retriever=vectordb.as_retriever(),
#                                   chain_type_kwargs={
#                                     "prompt": PromptTemplate(
#                                         template=template,
#                                         input_variables=["context", "summaries"],
#                                     ),
#                                   },
#                                   return_source_documents=True)







# llm functions
def process_llm_response(llm_response):
    print(llm_response)
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