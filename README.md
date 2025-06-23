# **QwenStack-RAG**

### **🧠 End-to-End Wikipedia QA with Qwen-Powered Retrieval-Augmented Generation**

## **📌 Project Description**

QwenStack-RAG is a modular, high-performance Retrieval-Augmented Generation (RAG) pipeline built entirely on the Qwen AI Stack. It showcases how to combine the power of:

Qwen3-Embedding-8B for dense semantic search,

Qwen3-Reranker-0.6B for intelligent passage reranking, and

Qwen3:4B LLM for coherent, fluent answer generation,

all orchestrated with LangChain, ChromaDB, Ollama and Gradio for seamless QA experiences.

Whether you're building an enterprise search engine or an intelligent assistant, QwenStack-RAG shows how to turn documents into conversational knowledge.


![Image](https://github.com/user-attachments/assets/fd58c829-11ef-4d39-9bf5-f4b15e04e18a)



## **Project Structure**

```

├── gradio_ui.py           # Gradio-powered web UI for interactive QA
├── ingest.py              # Embeds and stores Wikipedia documents into ChromaDB
├── qwen_reranker.py       # Implements Qwen3-Reranker-0.6B logic
├── stores/
│   └── Indian_Culture/    # Vector store holding Wikipedia embeddings
├── requirements.txt       # Required Python libraries and dependencies

```

## **🎯 Use Case**

1. Domain-specific QA: Extract relevant knowledge from a curated corpus (e.g., Indian Culture on Wikipedia)

2. Enhanced Answer Quality: Use intelligent reranking to improve context selection

3. RAG Benchmarking: Evaluate how embedding + reranking + generation models from Qwen integrate seamlessly

4. Developer Demo: Learn how to plug-and-play Qwen stack into your own LangChain-based pipelines


## **Installation Instructions**

Follow these simple steps to get the Voice to Text functionality running on your local machine:

1. Clone the repository to your local machine:
    ```bash
   git clone https://github.com/Ginga1402/QwenStack-RAG--Qwen3--Embeddings--Reranker.git
    ```
2. Navigate into the project directory:
    ```bash
    cd QwenStack-RAG--Qwen3--Embeddings--Reranker
    ```
3. Set up a virtual environment (recommended for Python projects):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
    ```
4. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## **Usage**

### 1. **Ingest Data**


```bash
python ingest.py
```

### 2. **Launch the UI**

Start the Gradio-powered QA interface.

```bash
python gradio_ui.py
```

### 3. **Ask Questions**

Open the Gradio link in your browser and ask domain-specific questions based on the Wikipedia data.

## **🧰 Technologies Used**

1. [Python](https://www.python.org/)

2. [Qwen3-Embedding-8B (for document embedding)](https://huggingface.co/Qwen/Qwen3-Embedding-8B)

3. [Qwen3-Reranker-0.6B (for context reranking)](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)

4. [Qwen3:4B (for answer generation)](https://huggingface.co/Qwen/Qwen3-4B)

5. [ChromaDB (for vector storage and retrieval)](https://www.trychroma.com/)

6. [LangChain (for RAG orchestration)](https://python.langchain.com/docs/introduction/)

7. [Gradio (for interactive UI)](https://www.gradio.app/)

8. [Ollama (for local model deployment)](https://ollama.com/)

9. [WikipediaLoader (for structured data loading)](https://python.langchain.com/docs/integrations/document_loaders/wikipedia/)



## **Contributing**
Contributions to this project are welcome! If you have ideas for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request.

## **License**
This project is licensed under the MIT License - see the LICENSE file for details.


