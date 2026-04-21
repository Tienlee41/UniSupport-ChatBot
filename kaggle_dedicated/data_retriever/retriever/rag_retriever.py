from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever
import torch
from ..schema import RagSource
from ..config import RagConfig
from .converter import RagSourceToDocumentConverter

class FaissRetriever:
    """Dense retrieval using FAISS + Sentence Transformers, with optional BM25 hybrid"""
    def __init__(self, config: RagConfig) -> None:
        self.config = config
        self.converter = RagSourceToDocumentConverter()
        device = (config.device or "cpu").lower()
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
            print("[FaissRetriever] CUDA requested but unavailable, falling back to CPU")
        self.embedding = HuggingFaceEmbeddings(
            model_name=config.embedding_name, 
            model_kwargs={"device": device}
        )

    def retrieve(self, sources: list[RagSource], query: str, k: int) -> list[RagSource]:
        if not sources:
            return []
        
        docs = [self.converter.convert(s) for s in sources]
        lookup = {s["chunk_index"]: s for s in sources}
        
        # Dense retrieval (FAISS)
        faiss_store = FAISS.from_documents(docs, self.embedding)
        dense_results = faiss_store.as_retriever(search_kwargs={"k": min(k * 2, len(sources))}).invoke(query)
        dense_chunks = [self.converter.revert(c) for c in dense_results]
        
        if not self.config.use_hybrid:
            results = dense_chunks[:k]
        else:
            # Sparse retrieval (BM25)
            bm25 = LangChainBM25Retriever.from_documents(docs)
            bm25.k = min(k * 2, len(sources))
            sparse_results = bm25.invoke(query)
            sparse_chunks = [self.converter.revert(c) for c in sparse_results]
            
            # Reciprocal Rank Fusion
            scores: dict[int, float] = {}
            alpha = self.config.hybrid_alpha
            for rank, c in enumerate(dense_chunks):
                scores[c["chunk_index"]] = scores.get(c["chunk_index"], 0) + alpha / (rank + 60)
            for rank, c in enumerate(sparse_chunks):
                scores[c["chunk_index"]] = scores.get(c["chunk_index"], 0) + (1 - alpha) / (rank + 60)
            
            sorted_idx = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
            results = [lookup[i] for i in sorted_idx]
        
        if self.config.keep_order:
            results = sorted(results, key=lambda s: s["chunk_index"])
        return results


class BM25Retriever:
    """Pure BM25 sparse retrieval"""
    def __init__(self, config: RagConfig) -> None:
        self.config = config
        self.converter = RagSourceToDocumentConverter()

    def retrieve(self, sources: list[RagSource], query: str, k: int) -> list[RagSource]:
        if not sources:
            return []
        docs = [self.converter.convert(s) for s in sources]
        retriever = LangChainBM25Retriever.from_documents(docs)
        retriever.k = k
        results = [self.converter.revert(c) for c in retriever.invoke(query)]
        if self.config.keep_order:
            results = sorted(results, key=lambda s: s["chunk_index"])
        return results
