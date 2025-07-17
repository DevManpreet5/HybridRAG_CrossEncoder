import os

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

texts = {
    "deep_learning.txt": """
Deep learning is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called artificial neural networks. 
It has revolutionized areas like computer vision, natural language processing, and speech recognition.
One of the major breakthroughs in deep learning came with the introduction of convolutional neural networks (CNNs) for image-related tasks.
""",

    "llms.txt": """
Large Language Models (LLMs) like GPT and Mixtral-8x7B-Instruct are pre-trained on vast corpora of text and can perform a variety of tasks such as summarization, translation, and question answering.
These models rely on the transformer architecture and are fine-tuned for specific tasks.
Retrieval-Augmented Generation (RAG) is a popular approach to improve factual correctness by combining retrieval techniques with LLMs.
""",

    "hybrid_rag.txt": """
A hybrid RAG system combines symbolic retrieval (like BM25) with neural dense retrieval and re-ranking using models like CrossEncoder.
This hybrid approach leverages the precision of symbolic methods and the contextual awareness of neural models.
It is particularly useful in enterprise applications where document relevance and correctness are critical.
"""
}

for filename, content in texts.items():
    with open(os.path.join(data_dir, filename), "w", encoding="utf-8") as f:
        f.write(content)

data_dir
