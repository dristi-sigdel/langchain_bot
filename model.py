import os
from getpass import getpass
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


loader = PyPDFDirectoryLoader('Data')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 256, chunk_overlap = 20)
text_chunks = text_splitter.split_documents(data)

# huggingface_hub_API_TOKEN = getpass()
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

vector_store = FAISS.from_documents(text_chunks, embedding = embeddings)
print(type(vector_store))

repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm = HuggingFaceHub(repo_id = repo_id, model_kwargs = {"temperature": 0.5, "max_length": 1024})

def load_llm(query):
    chain = RetrievalQA.from_chain_type(llm = llm, chain_type="stuff", retriever = vector_store.as_retriever(search_kwargs={"k":2}))
    prompt = query
    answer = chain({"query": prompt}, return_only_outputs= True)
    return answer