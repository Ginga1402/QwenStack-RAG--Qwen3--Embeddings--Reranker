import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

print("*"*100)
print(torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch {torch.__version__} ({DEVICE})")
print("*"*100)



modelname = "Qwen/Qwen3-Embedding-0.6B"
modelkwargs = {'device': 'cuda'}

embeddings = HuggingFaceEmbeddings(model_name=modelname,
                                    model_kwargs=modelkwargs
                                    )

docs = WikipediaLoader(query="India").load()
print("Lenght of Docs" + str(len(docs)))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(docs)


vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/Indian_Culture")

print("Vector Store Created.......")