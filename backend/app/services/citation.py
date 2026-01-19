"""
Citation verification service for validating LLM claims against sources
"""
import logging
import re
from typing import List, Dict, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class CitationService:
    """Service for verifying citations in LLM responses"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def verify_citations(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify that citations in the response are grounded in sources
        
        Args:
            response: LLM-generated response text
            sources: List of source chunks used for generation
            
        Returns:
            Verification results with any issues found
        """
        logger.info("Verifying citations in response")
        
        # Extract citation references from response
        citation_refs = self._extract_citations(response)
        
        if not citation_refs:
            logger.warning("No citations found in response")
            return {
                "passed": False,
                "issues": [{
                    "type": "no_citations",
                    "description": "Response contains no citations"
                }]
            }
        
        issues = []
        
        # Check each citation
        for source_num in citation_refs:
            # Check if source number exists
            if source_num > len(sources):
                issues.append({
                    "source_number": source_num,
                    "issue_type": "hallucinated",
                    "description": f"Source {source_num} does not exist (only {len(sources)} sources available)"
                })
                continue
            
            # Get the source content
            source = sources[source_num - 1]  # 0-indexed
            source_content = source.get("content", "")
            
            # Extract claims attributed to this source
            claims = self._extract_claims_for_source(response, source_num)
            
            # Verify each claim against source content
            for claim in claims:
                if not self._verify_claim_in_source(claim, source_content):
                    issues.append({
                        "source_number": source_num,
                        "issue_type": "misrepresented",
                        "description": f"Claim not found in source: '{claim[:100]}...'"
                    })
        
        # Check for claims without citations
        uncited_claims = self._find_uncited_claims(response, citation_refs)
        if uncited_claims:
            issues.append({
                "issue_type": "missing_citations",
                "description": f"Found {len(uncited_claims)} potentially uncited claims"
            })
        
        passed = len(issues) == 0
        
        logger.info(f"Citation verification: {'PASSED' if passed else 'FAILED'} ({len(issues)} issues)")
        
        return {
            "passed": passed,
            "issues": issues,
            "total_citations": len(citation_refs),
            "total_sources": len(sources)
        }
    
    def _extract_citations(self, text: str) -> List[int]:
        """
        Extract citation references like [Source 1], [Source 2] from text
        
        Returns list of source numbers
        """
        pattern = r'\[Source\s+(\d+)\]'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return sorted(set(int(m) for m in matches))
    
    def _extract_claims_for_source(self, text: str, source_num: int) -> List[str]:
        """
        Extract claims attributed to a specific source
        
        Looks for patterns like:
        - "According to [Source N], ..."
        - "[Source N] states that ..."
        - "As mentioned in [Source N], ..."
        """
        claims = []
        
        # Pattern to match sentences with citations
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if re.search(rf"\[Source\s+{source_num}\]", sentence, re.IGNORECASE):
                # Remove the citation marker for verification
                claim = re.sub(rf"\[Source\s+{source_num}\]", "", sentence, flags=re.IGNORECASE).strip()
                if claim:
                    claims.append(claim)
        
        return claims
    
    def _verify_claim_in_source(self, claim: str, source_content: str) -> bool:
        """
        Verify if a claim is supported by the source content
        
        Uses fuzzy matching to account for paraphrasing
        """
        # Normalize text for comparison
        claim_lower = claim.lower()
        source_lower = source_content.lower()
        
        # Remove common prefixes
        prefixes = [
            "according to",
            "as mentioned",
            "states that",
            "indicates that",
            "suggests that",
            "shows that"
        ]
        
        for prefix in prefixes:
            if claim_lower.startswith(prefix):
                claim_lower = claim_lower[len(prefix):].strip()
        
        # Extract key terms from claim (nouns, numbers, proper nouns)
        key_terms = self._extract_key_terms(claim_lower)
        
        if not key_terms:
            return True

        hits = sum(1 for term in key_terms if term in source_lower)
        
        if len(key_terms) <= 4:
            return hits >= 1
        elif len(key_terms) <= 10:
            return hits >= 2
        else:
            return hits >= 3
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text (simplified version)
        
        In production, use spaCy or similar NLP library for better accuracy
        """
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        junk = {"source", "context", "question", "answer"}
        
        # Split into words and filter
        words = re.findall(r'\b\w+\b', text.lower())
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        key_terms = [w for w in key_terms if w not in junk]
        
        # Also extract numbers and multi-word phrases
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        key_terms.extend(numbers)
        
        return key_terms
    
    def _find_uncited_claims(self, text: str, cited_sources: List[int]) -> List[str]:
        """
        Find factual claims that lack citations
        
        This is a heuristic approach - identifies sentences with specific patterns
        that typically indicate factual claims
        """
        uncited = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        
        allowed_no_citation_phrases = {
            "not found in the provided documents",
            "not available in the provided documents",
            "insufficient information in the provided documents",
            "i don't have enough information in the provided documents",
        }
        
        for sentence in sentences:
            if not sentence.strip():
                continue

            s_lower = sentence.strip().lower().rstrip(".")
            if s_lower in allowed_no_citation_phrases:
                continue
            # if sentence has no citation marker, flag it
            if not re.search(r'\[Source\s+\d+\]', sentence, re.IGNORECASE):
                uncited.append(sentence.strip())

        return uncited
    
    async def get_citation_statistics(self) -> Dict[str, Any]:
        """Get statistics about citation usage across the system"""
        
        # Most cited chunks
        result = await self.db.execute(text("""
            SELECT 
                c.id,
                c.content,
                c.citation_count,
                r.filename
            FROM chunks c
            JOIN resources r ON c.resource_id = r.id
            WHERE c.citation_count > 0
            ORDER BY c.citation_count DESC
            LIMIT 10
        """))
        
        most_cited = [
            {
                "chunk_id": str(row.id),
                "filename": row.filename,
                "citation_count": row.citation_count,
                "preview": row.content[:200] + "..."
            }
            for row in result.fetchall()
        ]
        
        return {
            "most_cited_chunks": most_cited
        }