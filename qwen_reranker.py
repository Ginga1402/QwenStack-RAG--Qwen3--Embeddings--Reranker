import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.schema import Document

print("*"*100)
print(torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch {torch.__version__} ({DEVICE})")
print("*"*100)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')



####### Use GPU
# model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen3-Reranker-0.6B",
#     torch_dtype=torch.float16,
#     attn_implementation="flash_attention_2"
# ).cuda().eval()


####### Use CPU
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-Reranker-0.6B",
    torch_dtype=torch.float16
)


# Token IDs for reranking
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

# Special tokens and config
max_length = 8192
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be &quot;yes&quot; or &quot;no&quot;.<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

def process_inputs(pairs):
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation='longest_first',
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()  # probability of "yes"
    return scores

# def rerank_documents(user_query, documents, instruction=None):
#     pairs = [format_instruction(instruction, user_query, doc) for doc in documents]
#     inputs = process_inputs(pairs)
#     scores = compute_logits(inputs)
#     reranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
#     return [doc for doc, _ in reranked]

######## Optional: only if you want to use Langchain Document

def rerank_documents(user_query, documents, instruction=None, k=3):
    """
    Re-rank a list of LangChain Document objects using the Qwen3 reranker model.

    Args:
        user_query (str): The user's input query.
        documents (List[Document]): A list of Document objects to rerank.
        instruction (str, optional): A task description.
        k (int): Number of top documents to return.

    Returns:
        List[Document]: Top-k reranked Document objects.
    """
    # Convert Documents to text
    doc_texts = [doc.page_content for doc in documents]

    # Format inputs
    pairs = [format_instruction(instruction, user_query, doc_text) for doc_text in doc_texts]
    inputs = process_inputs(pairs)

    # Compute relevance scores
    scores = compute_logits(inputs)

    # Rerank and select top-k
    reranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    top_k_documents = [doc for doc, _ in reranked[:k]]

    return top_k_documents



##### Try yourself!

# query = "What is the capital of China?"
# docs = [
#     "The capital of China is Beijing.",
#     "Shanghai is a major city in China.",
#     "China is a country in East Asia.",
#     "The Forbidden City is located in Beijing, which is the capital of China."
# ]
# result = rerank_documents(query, docs)
# print("Reranked Documents:\n", result)



######################################

# query = "What are the historical names of India?"
# docs = [
#     Document(page_content="India is also called Bharat and Hindustan."),
#     Document(page_content="The Himalayas are located in North India."),
#     Document(page_content="India has a rich cultural heritage.")
# ]

# top_docs = rerank_documents(query, docs, k=2)
# for doc in top_docs:
#     print(doc.page_content)


