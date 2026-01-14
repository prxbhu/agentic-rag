"""
Enhanced Production RAG Services
- Advanced Reranking (Cohere, Cross-Encoder, MMR)
- Circuit Breakers & Retries
- Query Decomposition & Self-Reflection
- Validation
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from langchain_core.messages import HumanMessage

from app.config import settings
from app.services.embedding import embedding_service

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"
    
    def call(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "open":
                if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                    self.state = "half_open"
                    logger.info(f"Circuit breaker half-open for {func.__name__}")
                else:
                    raise Exception(f"Circuit breaker open for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                if self.state == "half_open":
                    self.state = "closed"
                    self.failures = 0
                return result
            except Exception as e:
                self.failures += 1
                self.last_failure_time = datetime.now()
                if self.failures >= self.failure_threshold:
                    self.state = "open"
                    logger.error(f"Circuit breaker opened for {func.__name__}")
                raise e
        return wrapper

async def retry_with_backoff(func, max_retries=3, initial_delay=1.0, backoff_factor=2.0):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1: raise
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            await asyncio.sleep(delay)
            delay *= backoff_factor

class AdvancedRerankingService:
    """
    Reranks search results to improve precision.
    Crucial for fixing cases where relevant docs are found but ranked low.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        method: str = "hybrid", 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        try:
            # If the result set is small, cross-encoder is most accurate
            # For larger sets, or if configured, use hybrid/MMR
            if method == "cross_encoder":
                return await self._cross_encoder_rerank(query, documents, top_k)
            elif method == "hybrid":
                return await self._hybrid_rerank(query, documents, top_k)
            else:
                return documents[:top_k]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]

    async def _cross_encoder_rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int):
        """Uses a high-precision Cross-Encoder model (slower but very accurate)"""
        try:
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
            
            pairs = [[query, doc["content"]] for doc in documents]
            scores = model.predict(pairs)
            
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
            
            reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]
        except Exception as e:
            logger.warning(f"Cross-encoder error (model might not be downloaded): {e}")
            return documents[:top_k]

    async def _hybrid_rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int):
        """Combines original rank with MMR (diversity) to surface buried results"""
        try:
            # 1. MMR Score (Diversity & Relevance)
            mmr_docs = await self._mmr_rerank(query, documents, len(documents))
            mmr_rank_map = {doc["chunk_id"]: i for i, doc in enumerate(mmr_docs)}
            
            combined_docs = []
            for i, doc in enumerate(documents):
                # Original semantic rank (normalized)
                original_score = 1.0 - (i / len(documents))
                
                # MMR rank (normalized)
                mmr_rank = mmr_rank_map.get(doc["chunk_id"], len(documents))
                mmr_score = 1.0 - (mmr_rank / len(documents))
                
                # Weighted Score: 40% Original, 60% MMR
                # We give higher weight to MMR/Diversity to bubble up specific answers
                final_score = (original_score * 0.4) + (mmr_score * 0.6)
                doc["combined_rerank_score"] = final_score
                combined_docs.append(doc)
            
            return sorted(combined_docs, key=lambda x: x["combined_rerank_score"], reverse=True)[:top_k]
        except Exception as e:
            logger.error(f"Hybrid rerank failed: {e}")
            return documents[:top_k]

    async def _mmr_rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int, lambda_mult=0.7):
        """Maximal Marginal Relevance"""
        if len(documents) == 0: return []
        
        try:
            query_embedding = embedding_service.embed_text(query)
            
            # Use cached embeddings or generate
            doc_embeddings = []
            valid_docs = []
            for doc in documents:
                if "embedding" in doc and doc["embedding"]:
                    doc_embeddings.append(doc["embedding"])
                    valid_docs.append(doc)
                else:
                    # Fallback generation (truncated for speed)
                    try:
                        emb = embedding_service.embed_text(doc["content"][:1000])
                        doc_embeddings.append(emb)
                        valid_docs.append(doc)
                    except: continue
            
            if not doc_embeddings: return documents[:top_k]

            # Calculation
            selected_indices = []
            remaining_indices = list(range(len(valid_docs)))
            
            # Select first doc
            sims = [self._cosine_sim(query_embedding, emb) for emb in doc_embeddings]
            if not sims: return documents[:top_k]
            
            best_idx = remaining_indices[np.argmax(sims)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            while len(selected_indices) < top_k and remaining_indices:
                mmr_scores = []
                for idx in remaining_indices:
                    curr_sim = self._cosine_sim(query_embedding, doc_embeddings[idx])
                    max_sim_to_selected = max([self._cosine_sim(doc_embeddings[idx], doc_embeddings[sel]) for sel in selected_indices])
                    mmr = (lambda_mult * curr_sim) - ((1 - lambda_mult) * max_sim_to_selected)
                    mmr_scores.append(mmr)
                
                best_idx = remaining_indices[np.argmax(mmr_scores)]
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            return [valid_docs[i] for i in selected_indices]
        except ImportError:
            logger.warning("Numpy not found, skipping MMR")
            return documents[:top_k]
        except Exception as e:
            logger.error(f"MMR error: {e}")
            return documents[:top_k]

    @staticmethod
    def _cosine_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class QueryDecompositionAgent:
    def __init__(self, llm):
        self.llm = llm

    async def decompose_query(self, query: str) -> List[str]:
        prompt = f"""Break down this question into 2-3 simpler sub-questions to help answer it accurately.
        If the question is simple, just return the original question.
        
        Question: {query}
        
        Sub-questions (one per line, no numbering):"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            subs = [line.strip() for line in response.content.split('\n') if line.strip()]
            return subs if subs else [query]
        except:
            return [query]

    async def self_reflect(self, query, answer, sources):
        prompt = f"""Rate this answer from 0-10 based on how well it answers the user's question using the provided context.
        Question: {query}
        Answer: {answer}
        Sources Used: {len(sources)}
        
        Format:
        SCORE: X
        SUGGESTIONS: ...
        """
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content
            score = 5
            for line in content.split('\n'):
                if "SCORE:" in line:
                    try:
                        score = int(line.split(":")[1].strip())
                    except: pass
            return {"score": score, "analysis": content}
        except:
            return {"score": 5, "analysis": "Could not analyze"}

class ValidationService:
    @staticmethod
    async def validate_search_results(results, min_score=0.25):
        if not results: return [], "No results"
        filtered = [r for r in results if r.get("combined_score", 0) >= min_score]
        # Always return at least top 3 if everything was filtered, to avoid "No info" responses
        if not filtered and results:
            return results[:3], "Low confidence matches"
        return filtered, ""

    @staticmethod
    async def validate_context_window(context, max_tokens=4000):
        # Simple word count approximation
        if len(context.split()) * 1.3 > max_tokens:
            return False, 0, "Context too long"
        return True, 0