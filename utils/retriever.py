import os
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=".chroma_store", embedding_function=embedding_model)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class HybridRetriever:
    def __init__(self):
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        self.bm25 = None
        self.texts = []

    def index(self, chunks):
        self.texts = chunks
        tokenized = [text.lower().split() for text in chunks]
        self.bm25 = BM25Okapi(tokenized)
        docs = [Document(page_content=chunk, metadata={"id": i}) for i, chunk in enumerate(chunks)]
        vectorstore.add_documents(docs)

    def retrieve(self, query, k=5, return_scores=False):
        bm25_hits = self.bm25.get_top_n(query.lower().split(), self.texts, n=k)
        docs = vectorstore.similarity_search(query, k=k)
        vector_hits = [doc.page_content for doc in docs]
        candidates = list(set(bm25_hits + vector_hits))

        pairs = [(query, text) for text in candidates]
        scores = self.cross_encoder.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
        if return_scores:
            return reranked[:k]  
        else:
            return [text for text, _ in reranked[:k]]


retriever = HybridRetriever()