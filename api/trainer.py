import os
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('api_key')

directory = 'database'
embedding = OpenAIEmbeddings()

loader = DirectoryLoader('pdf', glob="./*.pdf", show_progress=True)
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

print("text", texts[15])
print("length: ", len(texts))


# Creating the vector db
vectordb = Chroma.from_documents(documents=texts, persist_directory=directory, embedding=embedding)
print("done")

vectordb.persist()
vectordb = None
