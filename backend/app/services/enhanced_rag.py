"""
Enhanced Production RAG Services - FIXED VERSION
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


W_RELEVANCE = 0.7  
W_DIVERSITY = 0.3 

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
        top_k: int = 15
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        try:
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
        """Uses a high-precision Cross-Encoder model"""
        try:
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
            
            pairs = [[query, doc["content"]] for doc in documents]
            scores = model.predict(pairs)
            
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
            
            reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]
        except Exception as e:
            logger.warning(f"Cross-encoder error: {e}")
            return documents[:top_k]

    async def _hybrid_rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int):
        try:
            if not documents:
                return []
            
            # Expand candidate pool for better diversity
            candidate_k = min(len(documents), max(top_k * 3, 30))
            candidates = documents[:candidate_k]
            
            # Normalize original scores
            raw_scores = []
            for d in candidates:
                try:
                    raw_scores.append(float(d.get("combined_score", 0.0)))
                except Exception:
                    raw_scores.append(0.0)

            s_min, s_max = min(raw_scores), max(raw_scores) 
            
            def norm(x: float) -> float:
                if s_max == s_min:
                    return 0.5
                return (x - s_min) / (s_max - s_min)
            
            # Get MMR diversified ranking
            mmr_docs = await self._mmr_rerank(query, candidates, len(candidates))
            mmr_rank_map = {
                str(doc.get("chunk_id")): i
                for i, doc in enumerate(mmr_docs)
                if doc.get("chunk_id") is not None
            }
            
            # Combine scores using proper weights
            combined_docs = []
            n = len(candidates)
            for i, doc in enumerate(candidates):
                chunk_id = str(doc.get("chunk_id", ""))

                # Normalized relevance from original search
                original_score = norm(float(doc.get("combined_score", 0.0)))

                # Diversity score from MMR ranking
                mmr_rank = mmr_rank_map.get(chunk_id, n - 1)
                diversity_score = 1.0 - (mmr_rank / max(n - 1, 1))

                # Use defined constants W_RELEVANCE and W_DIVERSITY
                final_score = (original_score * W_RELEVANCE) + (diversity_score * W_DIVERSITY)

                new_doc = dict(doc)
                new_doc["combined_rerank_score"] = float(final_score)
                new_doc["normalized_relevance_score"] = float(original_score)
                new_doc["diversity_score"] = float(diversity_score)

                combined_docs.append(new_doc)
            
            combined_docs.sort(key=lambda x: x["combined_rerank_score"], reverse=True)

            # Return top_k plus remaining tail
            tail = [d for d in documents[candidate_k:]]
            return combined_docs[:top_k] + tail[:max(0, top_k - len(combined_docs))]
            
        except Exception as e:
            logger.error(f"Hybrid rerank failed: {e}")
            return documents[:top_k]

    async def _mmr_rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int, lambda_mult=0.7):
        """Maximal Marginal Relevance - promotes diversity"""
        if len(documents) == 0: 
            return []
        
        try:
            query_embedding = embedding_service.embed_text(query)
            
            # Collect embeddings
            doc_embeddings = []
            valid_docs = []
            for doc in documents:
                if "embedding" in doc and doc["embedding"]:
                    doc_embeddings.append(doc["embedding"])
                    valid_docs.append(doc)
                else:
                    # Generate embedding for content
                    try:
                        emb = embedding_service.embed_text(doc["content"][:1000])
                        doc_embeddings.append(emb)
                        valid_docs.append(doc)
                    except: 
                        continue
            
            if not doc_embeddings: 
                return documents[:top_k]

            # MMR algorithm
            selected_indices = []
            remaining_indices = list(range(len(valid_docs)))
            
            # Select first doc (most similar to query)
            sims = [self._cosine_sim(query_embedding, emb) for emb in doc_embeddings]
            if not sims: 
                return documents[:top_k]
            
            best_idx = remaining_indices[np.argmax(sims)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            # Iteratively select diverse documents
            while len(selected_indices) < top_k and remaining_indices:
                mmr_scores = []
                for idx in remaining_indices:
                    # Relevance to query
                    curr_sim = self._cosine_sim(query_embedding, doc_embeddings[idx])
                    # Max similarity to already selected docs
                    max_sim_to_selected = max([
                        self._cosine_sim(doc_embeddings[idx], doc_embeddings[sel]) 
                        for sel in selected_indices
                    ])
                    # MMR score balances relevance and diversity
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
        """
        Generate diverse sub-queries using different perspectives
        """
        prompt = f"""You are an expert at breaking down questions to improve information retrieval.

                Generate 3 DIVERSE search queries to find information about this question. Each query should:
                - Use DIFFERENT keywords and phrasings
                - Target a specific aspect (who, what, when, where, how much)
                - Be concrete and specific

                Original Question: {query}

                Generate 3 diverse search queries (one per line, no numbering or explanations):

                Example:
                If asked "What is the fee for GATE 2026?"
                Good queries:
                - GATE 2026 application fee cost
                - Graduate Aptitude Test Engineering 2026 registration charges
                - GATE exam fee structure payment details

                Your queries:"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            lines = response.content.strip().split('\n')
            
            # Clean up the response
            subs = []
            for line in lines:
                line = line.strip()
                # Skip empty lines, headers, examples
                if not line or line.startswith('Example') or line.startswith('If asked') or line.startswith('Good queries'):
                    continue
                # Remove numbering, bullets, dashes
                line = line.lstrip('0123456789.-) ')
                if line:
                    subs.append(line)
            
            # Take top 3, fallback to original if parsing failed
            return subs[:3] if len(subs) >= 2 else [query]
            
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            # Fallback: generate simple variations
            return [
                query,
                query.replace("fee", "cost"),
                query.replace("What is", "How much is")
            ][:3]

    async def self_reflect(self, query: str, answer: str, sources: List[Dict]) -> Dict:
        """
        IMPROVED: More rigorous evaluation with specific criteria
        """
        
        # Check if answer has citations
        has_citations = '[Source' in answer or 'Source 1' in answer
        source_count = len(sources)
        
        prompt = f"""Evaluate this RAG answer on a scale of 0-10.

                Question: {query}
                Answer: {answer}
                Number of sources available: {source_count}

                Scoring Rubric:
                1. Direct Answer (3 points): Does it directly answer the question?
                2. Source Support (3 points): Is every claim backed by sources?
                3. Completeness (2 points): Is the answer thorough?
                4. Citation Quality (2 points): Are citations present and correct?

                Deductions:
                - No citations: -3 points
                - Vague answer: -2 points
                - Missing key information: -2 points

                Format your response:
                SCORE: [0-10]
                REASONING: [Why this score?]
                MISSING: [What's missing if score < 8]
                """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content
            
            # Parse score
            score = 5  # Default middle score
            for line in content.split('\n'):
                if line.strip().upper().startswith('SCORE:'):
                    try:
                        score_text = line.split(':')[1].strip()
                        # Extract just the number (handle "8/10" or "8" formats)
                        score = int(score_text.split('/')[0].strip())
                        score = max(0, min(10, score))  # Clamp to 0-10
                    except:
                        pass
            
            # Auto-penalize if no citations
            if not has_citations and score > 5:
                score = min(score, 5)
                content += "\n[Auto-adjustment: No citations found, score capped at 5]"
            
            return {
                "score": score, 
                "analysis": content,
                "has_citations": has_citations
            }
            
        except Exception as e:
            logger.warning(f"Self-reflection failed: {e}")
            return {
                "score": 5 if has_citations else 3, 
                "analysis": "Could not analyze",
                "has_citations": has_citations
            }
class ValidationService:
    """
    IMPROVED: More lenient validation to avoid filtering out relevant docs
    """
    @staticmethod
    async def validate_search_results(results, min_score=0.15):
        """
        FIXED: Lowered threshold from 0.25 to 0.15 to capture more relevant docs
        """
        if not results: 
            return [], "No results"
        
        # Filter by minimum score
        filtered = [r for r in results if r.get("combined_score", 0) >= min_score]
        
        # IMPROVED: Always return top results even if below threshold
        if not filtered and results:
            top_3 = results[:3]
            return top_3, "Low confidence - returning top matches"
        
        return filtered, ""

    @staticmethod
    async def validate_context_window(context, max_tokens=4000):
        approx_tokens = len(context.split()) * 1.3
        if approx_tokens > max_tokens:
            return False, int(approx_tokens), "Context exceeds token limit"
        return True, int(approx_tokens), ""