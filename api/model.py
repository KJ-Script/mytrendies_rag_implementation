import os
from dotenv import load_dotenv
import sys
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('api_key')

directory = 'database'
embedding = OpenAIEmbeddings()


vectordb=Chroma(persist_directory=directory, embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

sys.modules[__name__] = process_llm_response

query = "give me a summary of the article Pumpkin Head aka Bobby along side with the Link to the article"
llm_response = qa_chain(query)
process_llm_response(llm_response)