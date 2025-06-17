from Logic import get_response
import gradio as gr

custom_css = """
.container {
    margin: 0 auto;
    padding: 15px;
    background-color: #1a1a1a;
}

.main-title {
    text-align: center;
    color: #00ff95;
    font-size: 2em;
    margin-bottom: 5px;
    font-weight: 600;
}

.subtitle {
    text-align: center;
    color: #b8b8b8;
    font-size: 1em;
    margin-bottom: 15px;
}

.tech-stack {
    background-color: #2d2d2d;
    padding: 12px;
    border-radius: 8px;
    margin: 10px 0;
    border: 1px solid #404040;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.tech-item {
    color: #e0e0e0;
    font-size: 0.9em;
    flex: 1 1 auto;
    min-width: 200px;
}

.input-box {
    background-color: #2d2d2d !important;
    border: 1px solid #404040 !important;
    color: #ffffff !important;
}

.output-box {
    background-color: #2d2d2d !important;
    border: 1px solid #404040 !important;
    color: #ffffff !important;
    padding: 12px !important;
    border-radius: 8px !important;
    font-size: 1em !important;
}

.section-header {
    color: #00ff95;
    font-size: 1.1em;
    margin: 10px 0 5px 0;
    font-weight: 500;
}

.description-text {
    color: #b8b8b8;
    line-height: 1.4;
    margin-bottom: 10px;
    font-size: 0.9em;
}

.custom-button {
    background-color: #00ff95 !important;
    color: #1a1a1a !important;
    border: none !important;
    padding: 8px 15px !important;
    border-radius: 5px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.custom-button:hover {
    background-color: #00cc78 !important;
    transform: translateY(-2px) !important;
}

.feature-box {
    background-color: #2d2d2d;
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
    border: 1px solid #404040;
    font-size: 0.9em;
}

.compact-row {
    margin: 5px 0 !important;
    padding: 5px !important;
}
"""

def build_gradio_interface():
    with gr.Blocks(css=custom_css) as demo:
        with gr.Row(elem_classes=["compact-row"]):
            gr.HTML("""
                <div class="main-title">🤖 QwenStack-RAG: Wikipedia Retrieval-Augmented Generation with Qwen LLM, Embeddings & Reranker</div>
                <div class="subtitle">A LangChain-Powered Demo Integrating Qwen's Full-Stack AI with ChromaDB for Scalable Knowledge Retrieval</div>
            """)

        with gr.Row(elem_classes=["compact-row"]):
            with gr.Column(scale=1):
                gr.HTML("""
                    <div class="tech-stack">
                        <div class="tech-item">🔸 LangChain</div>
                        <div class="tech-item">🔸 Qwen3-Reranker-0.6B</div>
                        <div class="tech-item">🔸 Qwen3-Embedding-8B</div>
                        <div class="tech-item">🔸 Qwen3:4b-LLM</div>
                        <div class="tech-item">🔸 Chroma Vector Store</div>
                        <div class="tech-item">🔸 WikipediaLoader</div>
                    </div>
                """)

        with gr.Row(elem_classes=["compact-row"]):
            with gr.Column():
                input_text = gr.Textbox(
                    label="Your Question",
                    placeholder="Enter your question here...",
                    lines=3,
                    elem_classes=["input-box"]
                )

        with gr.Row(elem_classes=["compact-row"]):
            submit_btn = gr.Button(
                "Get Answer",
                elem_classes=["custom-button"]
            )

        with gr.Row(elem_classes=["compact-row"]):
            output_text = gr.Textbox(
                label="AI Response",
                lines=4,
                elem_classes=["output-box"]
            )

        with gr.Row(elem_classes=["compact-row"]):
            with gr.Column():
                gr.HTML("""
                    <div class="feature-box">
                        <div style="color: #00ff95; margin-bottom: 5px;">System Features:</div>
                        <div style="color: #e0e0e0; display: flex; flex-wrap: wrap; gap: 10px;">
                            <span>✦ QwenStack-RAG is a Wikipedia-based Retrieval-Augmented Generation (RAG) demo built using the full Qwen AI stack, including the Qwen LLM, embedding model, and reranker, all integrated with LangChain and ChromaDB. It demonstrates how to build a performant, modular RAG pipeline by combining dense retrieval, intelligent reranking, and fluent response generation. This project highlights Qwen’s capabilities in end-to-end knowledge retrieval and serves as a practical reference for developers building RAG systems with Qwen and LangChain.</span>
                    </div>
                """)
        # Connect the input and output
        submit_btn.click(
            fn=get_response,
            inputs=[input_text],
            outputs=[output_text]
        )

    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = build_gradio_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,

    )
