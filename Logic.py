from langchain import PromptTemplate, LLMChain
import os
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from io import BytesIO
from langchain.llms import CTransformers
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import torch

from qwen_reranker import rerank_documents

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



llm = Ollama(base_url = 'http://localhost:11434',model = 'qwen3:4b',temperature = 0.3,keep_alive = -1) #'qwen2.5:3b



#### Use Ctransformers 

# local_llm = "llama-2-13b-chat.Q4_K_M.gguf"

# config = {
# 'max_new_tokens': 1024,
# 'repetition_penalty': 1.1,
# 'temperature': 0.1,
# 'top_k': 50,
# 'top_p': 0.9,
# 'stream': True,
# 'threads': int(os.cpu_count() / 2)
# }

# llm = CTransformers(
#     model=local_llm,
#     model_type="mistral",
#     lib="avx2", #for CPU use
#     **config
# )



print("Model's Initialized...")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
load_vector_store = Chroma(persist_directory="stores/Indian_Culture", embedding_function=embeddings)
# retriever = load_vector_store.as_retriever(search_kwargs={"k":5})
retriever = load_vector_store.as_retriever(search_kwargs={"k":5})
chain_type_kwargs = {"prompt": prompt}



def get_response(input):
    query = input
    print("The query asked is: ",query)
    retrieve_data = retriever.invoke(query)
    print("Retrieved Data is: ")
    print(retrieve_data)
    print("#"*50)
    rerank_chunks = rerank_documents(query, retrieve_data, k=3)
    print("Reranked Chunks are: ")
    print(rerank_chunks)
    print("#"*50)
    chain_type_kwargs = {"prompt": prompt}
    reranked_vector_store = Chroma.from_documents(rerank_chunks, embeddings, collection_metadata={"hnsw:space": "cosine"}, collection_name="rerank_retrieval")
    load_reranked_vector_store = Chroma(collection_name = "rerank_retrieval" , embedding_function=embeddings)
    reranked_retriever = load_reranked_vector_store.as_retriever(search_kwargs={"k":3})
    
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=reranked_retriever, 
                                    return_source_documents=True, 
                                    chain_type_kwargs=chain_type_kwargs, 
                                    verbose=True)
    response = qa(query)
    print("response is :",response)
    result = response['result']
    return result
    # return rerank_chunks

#qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_kwargs={"k": 2}))





##### Try yourself!

# sample = get_response("tell me about the rich heritage of India")
# print("query: " , sample['query'])
# print("response: ", sample['result'])
# print("source documents: ", sample['source_documents'])







            