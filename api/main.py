# os and langchain imports
import os
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory

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




load_dotenv()

# initalizing key and embeddings
os.environ["OPENAI_API_KEY"] = os.getenv('api_key')
directory = 'database'
embedding = OpenAIEmbeddings()


# initalizing vector databases and retrievers
vectordb=Chroma(persist_directory=directory, embedding_function=embedding)
retriever = vectordb.as_retriever()


template = """
You are an AI assistant tasked with helping users of a the blogging website MyTrendingStories.com
When you are greeted, respond back with a greeting. Reply to greetings as the MyTrendingStories chat bot.
You are trained on an assortment of articles and blogs taken from this blogging site. This will be your context.
Your task is to summarise, recommend and assist users using ONLY the context and history provided. Be sure to list the Source/Link/HyperLink of the article refrenced.
If you are asked to recommend articles, blogs and things of that sorts, your recommendation should only be from the context provided. Be sure to list the Source/Link/HyperLink of the article refrenced.
If you are asked to summarise an article, summarise the article only from the context provided. Be sure to list the Source/Link/HyperLink of the article refrenced.
If you are asked about an article, respond using only the context provided. Be sure to list the Source/Link/HyperLink of the article refrenced.

In all of your responses, ALWAYS include the Link/Hyperlink/Source of the article or articles you refrenced.

In all of your responses, ALWAYS include the Author or Written by section of the article or articles you refrenced.
Do not include the link of any article that is not from MyTrendingStories

You must respond in the language you were prompted in. If you were prompted in french, respond in french. If you were prompted in German, respond in German etcetera. 
When responding the Link/Hyperlink/Source section should always be in English

{context}

------------------------------

History: {history}

------------------------------

Question: {question}"""

PROMPT = PromptTemplate(
    template=template, input_variables=["history", "context", "question"]
)


# changing retriever
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                  chain_type="stuff", 
                                  retriever=retriever,
                                  chain_type_kwargs={"prompt": PROMPT, "memory": ConversationBufferMemory(memory_key="history", input_key="question")}, 
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