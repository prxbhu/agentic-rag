"""
Multi-factor ranking service for re-ranking search results
"""
import logging
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.config import settings

logger = logging.getLogger(__name__)


class RankingService:
    """Service for multi-factor ranking of search results"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def rank_results(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Apply multi-factor ranking to search results
        
        Factors:
        - Base relevance (40%): from hybrid search score
        - Citation frequency (15%): how often chunk has been cited
        - Recency (15%): how recent the document is
        - Specificity (15%): how specific/detailed the chunk is
        - Source quality (15%): overall quality score of the document
        
        Args:
            results: List of search results
            query: Original query
            
        Returns:
            Ranked list of results with final scores
        """
        if not results:
            return []
        
        logger.info(f"Ranking {len(results)} results")
        
        # Enrich results with additional metadata
        enriched_results = await self._enrich_results(results)
        
        # Calculate individual factor scores
        for result in enriched_results:
            scores = self._calculate_factor_scores(result, enriched_results)
            result.update(scores)
            
            # Calculate final weighted score
            result["final_score"] = (
                scores["base_relevance"] * settings.BASE_RELEVANCE_WEIGHT +
                scores["citation_frequency"] * settings.CITATION_FREQ_WEIGHT +
                scores["recency_score"] * settings.RECENCY_WEIGHT +
                scores["specificity_score"] * settings.SPECIFICITY_WEIGHT +
                scores["source_quality"] * settings.SOURCE_QUALITY_WEIGHT
            )
        
        # Sort by final score
        ranked_results = sorted(
            enriched_results,
            key=lambda x: x["final_score"],
            reverse=True
        )
        
        # Detect and flag conflicting sources
        ranked_results = self._detect_conflicts(ranked_results)
        
        logger.info(f"Ranking complete. Top score: {ranked_results[0]['final_score']:.3f}")
        
        return ranked_results
    
    async def _enrich_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich results with additional metadata from database"""
        enriched = []
        
        for result in results:
            # Get resource metadata
            sql = text("""
                SELECT 
                    r.created_at,
                    r.metadata,
                    COALESCE(sq.quality_score, 0.5) as quality_score,
                    COALESCE(sq.specificity_score, 0.5) as specificity_score
                FROM resources r
                LEFT JOIN source_quality sq ON r.id = sq.resource_id
                WHERE r.id = :resource_id
            """)
            
            db_result = await self.db.execute(
                sql,
                {"resource_id": str(result["resource_id"])}
            )
            row = db_result.fetchone()
            
            if row and row.created_at:
                result["resource_created_at"] = row.created_at
                result["resource_metadata"] = row.metadata
                result["stored_quality_score"] = float(row.quality_score)
                result["stored_specificity_score"] = float(row.specificity_score)
            else:
                result["resource_created_at"] =  datetime.now()
                result["resource_metadata"] = {}
                result["stored_quality_score"] = 0.5
                result["stored_specificity_score"] = 0.5
            
            enriched.append(result)
        
        return enriched
    
    def _calculate_factor_scores(
        self,
        result: Dict[str, Any],
        all_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate individual factor scores for a result"""
        
        # 1. Base relevance (normalized from hybrid search score)
        base_relevance = result.get("combined_score", 0.5)
        
        # 2. Citation frequency (normalized)
        citation_count = result.get("citation_count", 0)
        max_citations = max(r.get("citation_count", 0) for r in all_results) or 1
        citation_frequency = min(citation_count / max_citations, 1.0)
        
        # 3. Recency score (exponential decay)
        created_at = result.get("resource_created_at") or datetime.now()
        recency_score = self._calculate_recency_score(created_at)
        
        # 4. Specificity score (from stored value or estimated)
        specificity_score = result.get("stored_specificity_score", 0.5)
        
        # Adjust based on chunk length and content density
        content_length = len(result.get("content", ""))
        if content_length > 1000:
            specificity_score = min(specificity_score + 0.1, 1.0)
        elif content_length < 200:
            specificity_score = max(specificity_score - 0.1, 0.0)
        
        # 5. Source quality
        source_quality = result.get("stored_quality_score", 0.5)
        
        return {
            "base_relevance": base_relevance,
            "citation_frequency": citation_frequency,
            "recency_score": recency_score,
            "specificity_score": specificity_score,
            "source_quality": source_quality
        }
    
    def _calculate_recency_score(self, created_at: datetime) -> float:
        """
        Calculate recency score with exponential decay
        
        Score decays to 0.5 after 1 year, continues decaying thereafter
        """
        now = datetime.now()
        if created_at.tzinfo is not None:
            created_at = created_at.replace(tzinfo=None)
        days_old = max((now - created_at).days, 0)
        
        # Exponential decay: score = e^(-days/365)
        # Clamp between 0.1 and 1.0
        import math
        score = math.exp(-days_old / 365.0)
        return max(0.1, min(1.0, score))
    
    def _detect_conflicts(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicting information between sources
        
        Uses simple heuristics:
        - Same topic but different numerical values
        - Contradictory keywords (increases vs. decreases, etc.)
        """
        # Group by topic similarity (top 5 results likely cover same topic)
        top_results = results[:5]
        
        # Look for contradictory patterns
        contradictory_patterns = [
            ("increase", "decrease"),
            ("rise", "fall"),
            ("growth", "decline"),
            ("positive", "negative"),
            ("successful", "failed"),
            ("approved", "rejected")
        ]
        
        for i, result in enumerate(top_results):
            content_lower = result["content"].lower()
            
            for pattern_a, pattern_b in contradictory_patterns:
                # Check if this result and any other top result contain contradictory terms
                if pattern_a in content_lower or pattern_b in content_lower:
                    for j, other in enumerate(top_results):
                        if i != j:
                            other_content = other["content"].lower()
                            if (pattern_a in content_lower and pattern_b in other_content) or \
                               (pattern_b in content_lower and pattern_a in other_content):
                                result["has_conflict"] = True
                                result["conflict_with"] = other["chunk_id"]
                                break
        
        return results
    
    async def update_citation_count(self, chunk_id: str):
        """Increment citation count for a chunk"""
        sql = text("""
            UPDATE chunks
            SET citation_count = citation_count + 1
            WHERE id = :chunk_id
        """)
        
        await self.db.execute(sql, {"chunk_id": chunk_id})
        await self.db.commit()
    
    async def update_source_quality(
        self,
        resource_id: str,
        quality_score: float = None,
        specificity_score: float = None
    ):
        """Update quality scores for a resource"""
        updates = []
        params = {"resource_id": resource_id}
        
        if quality_score is not None:
            updates.append("quality_score = :quality_score")
            params["quality_score"] = quality_score
        
        if specificity_score is not None:
            updates.append("specificity_score = :specificity_score")
            params["specificity_score"] = specificity_score
        
        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            
            sql = text(f"""
                INSERT INTO source_quality (resource_id, {', '.join(updates.split(' = ')[0::2])})
                VALUES (:resource_id, {', '.join([':' + p for p in params.keys() if p != 'resource_id'])})
                ON CONFLICT (resource_id) 
                DO UPDATE SET {', '.join(updates)}
            """)
            
            await self.db.execute(sql, params)
            await self.db.commit()