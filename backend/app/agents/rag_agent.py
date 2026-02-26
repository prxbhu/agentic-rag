"""
LangGraph-based RAG agent for orchestrating the retrieval pipeline
"""
import logging
from typing import TypedDict, List, Dict, Any, Annotated, AsyncGenerator
from uuid import UUID
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from app.config import settings
from app.services.enhanced_rag import AdvancedRerankingService, QueryDecompositionAgent, ValidationService, retry_with_backoff
from app.services.llm_service import LLMServiceFactory
from app.services.llama_index_service import init_llamaindex

logger = logging.getLogger(__name__)

class LLMResponse:
    """Ensures response.content exists for downstream logic"""
    def __init__(self, content: str):
        self.content = content
        
class RAGState(TypedDict):
    """State for the RAG agent"""
    # Input
    query: str
    workspace_id: UUID
    conversation_id: UUID
    
    # Intermediate states
    expanded_queries: List[str]
    search_results: List[Dict[str, Any]]
    reranked_results: List[Dict[str, Any]]
    ranked_results: List[Dict[str, Any]]
    context: str
    
    # Output
    response: str
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    context_sources: List[Dict[str, Any]]

    
    # Quality control
    self_reflection: Dict[str, Any]
    needs_refinement: bool
    refinement_count: int
    
    # Messages for chat history
    messages: Annotated[List, operator.add]


class RAGAgent:
    """LangGraph-based RAG agent"""
    
    def __init__(self, search_service, ranking_service, citation_service, llm_provider):
        self.search_service = search_service
        self.ranking_service = ranking_service
        self.citation_service = citation_service
        
        # Initialize enhanced services
        self.reranking_service = AdvancedRerankingService(search_service.db)
        self.validation_service = ValidationService()
        
        self.vector_store = init_llamaindex()
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        
        # Initialize LLM
        self.llm = LLMServiceFactory.create_llm_service(llm_provider)
        
        # Initialize query decomposition
        self.query_decomposer = QueryDecompositionAgent(self.llm)
        
        # Build the graph
        self.graph = self._build_graph()
          
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(RAGState)
        
         # Add nodes
        workflow.add_node("expand_query", self.expand_query)
        workflow.add_node("retrieve_data", self.retrieve_data)
        workflow.add_node("rerank_results", self.rerank_results)
        workflow.add_node("rank_results", self.rank_results)
        workflow.add_node("assemble_context", self.assemble_context)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("verify_citations", self.verify_citations)
        workflow.add_node("self_reflect", self.self_reflect)
        workflow.add_node("refine_response", self.refine_response)
        
        # Define edges
        workflow.set_entry_point("expand_query")
        workflow.add_edge("expand_query", "retrieve_data")
        workflow.add_edge("retrieve_data", "rerank_results")
        workflow.add_edge("rerank_results", "rank_results")
        workflow.add_edge("rank_results", "assemble_context")
        workflow.add_edge("assemble_context", "generate_response")
        workflow.add_edge("generate_response", "verify_citations")
        workflow.add_edge("verify_citations", "self_reflect")
        
        # Conditional: refine if needed (score < 7)
        workflow.add_conditional_edges(
            "self_reflect",
            lambda state: "refine" if state.get("needs_refinement") and state.get("refinement_count", 0) < 1 else "end",
            {
                "refine": "refine_response",
                "end": END
            }
        )
        
        workflow.add_edge("refine_response", "verify_citations")
        
        return workflow.compile()
    
    async def expand_query(self, state: RAGState) -> Dict:
        """Expand the user query into multiple variants"""
        logger.info(f"Expanding query: {state['query']}")
        
        try:
            sub_queries = await self.query_decomposer.decompose_query(state["query"])
            queries = [state["query"]] + [q for q in sub_queries if q != state["query"]]
            logger.info(f"Expanded to {len(queries)} queries")
            return {"expanded_queries": queries[:4], "messages": [HumanMessage(content=state["query"])]}
        
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return {
                "expanded_queries": [state["query"]],
                "messages": [HumanMessage(content=state["query"])]
            }
    
    async def hybrid_search(self, state: RAGState) -> Dict:
        """
        FIXED: Better deduplication strategy and result merging
        """
        logger.info(f"Performing hybrid search for {len(state['expanded_queries'])} queries")
        
        # Store results by chunk_id to track scores across queries
        chunk_results = {}  # chunk_id -> best result
        chunk_query_hits = {}  # chunk_id -> number of queries that found it
        
        async def run_one(q: str):
            return await self.search_service.hybrid_search(
                query=q,
                workspace_id=state["workspace_id"],
                limit=settings.MAX_SEARCH_RESULTS
            )

        for idx, q in enumerate(state["expanded_queries"]):
            try:
                results = await retry_with_backoff(lambda: run_one(q))
                logger.info(f"Query '{q[:50]}...' returned {len(results)} results")
            except Exception as e:
                logger.warning(f"Hybrid search failed for query='{q}': {e}")
                continue

            for r in results:
                cid = str(r["chunk_id"])
                
                # Track how many queries found this chunk (boost for multi-query hits)
                chunk_query_hits[cid] = chunk_query_hits.get(cid, 0) + 1
                
                # Keep the result with highest combined score
                if cid not in chunk_results or r.get("combined_score", 0) > chunk_results[cid].get("combined_score", 0):
                    chunk_results[cid] = r

        # Convert back to list and boost scores for chunks found by multiple queries
        all_results = []
        for cid, result in chunk_results.items():
            # Boost score if multiple queries found this chunk (indicates high relevance)
            query_boost = 1.0 + (0.1 * (chunk_query_hits[cid] - 1))  # +10% per additional query
            result["combined_score"] = result.get("combined_score", 0) * query_boost
            result["query_hit_count"] = chunk_query_hits[cid]
            all_results.append(result)
        
        # Sort by boosted score
        all_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        # More lenient validation
        validated, note = await ValidationService.validate_search_results(all_results, min_score=0.10)
        
        # If we got very few results, check if it's a data problem
        if len(validated) < 3:
            logger.warning(f"⚠️ Only {len(validated)} unique documents found. This suggests:")
            logger.warning("  1. Limited documents in workspace about this topic")
            logger.warning("  2. Query terms don't match document content")
            logger.warning("  3. Possible chunking/embedding issues")
            logger.warning("Consider uploading more documents or checking document content.")
        
        if note:
            logger.warning(f"Search validation note: {note}")
        
        logger.info(f"Found {len(validated)} unique search results after deduplication")
        logger.info(f"Top result score: {validated[0].get('combined_score', 0):.3f}" if validated else "No results")
        
        return {
            "search_results": validated[:50]  # Limit to top 50
        }

    async def retrieve_data(self, state: RAGState) -> Dict:
        """
        Replaces manual hybrid_search with LlamaIndex Retriever.
        Maintains your existing multi-query deduplication logic!
        """
        logger.info(f"Retrieving data using LlamaIndex for {len(state['expanded_queries'])} queries")
        logger.info(f"LlamaIndex Retrival - Target Workspace ID: {state['workspace_id']}")
        
        chunk_results = {}
        chunk_query_hits = {}
        
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="workspace_id", value=str(state["workspace_id"]))]
        )
        retriever = self.index.as_retriever(
            similarity_top_k=settings.MAX_SEARCH_RESULTS,
            filters=filters
        )

        async def run_one(q: str):
            logger.info(f"Executing LlamaIndex retrieval for: '{q}'")
            nodes = await retriever.aretrieve(q)
            logger.info(f"Query '{q}' returned {len(nodes)} nodes.")
            return nodes

        for idx, q in enumerate(state["expanded_queries"]):
            try:
                nodes = await retry_with_backoff(lambda: run_one(q))
            except Exception as e:
                logger.warning(f"Retrieval failed for query='{q}': {e}")
                continue

            for node in nodes:
                cid = str(node.node_id)
                chunk_query_hits[cid] = chunk_query_hits.get(cid, 0) + 1
                
                score = node.score if node.score is not None else 0.0
                logger.info(f"Retrieved Node: {cid[:8]}... | Score: {score:.3f} | File: {node.metadata.get('filename')}")
                
                result_dict = {
                    "chunk_id": cid,
                    "content": node.get_content(),
                    "combined_score": node.score or 0.0,
                    "metadata": node.metadata,
                    "filename": node.metadata.get("filename", "unknown"),
                    "resource_id": node.metadata.get("resource_id"),
                    "parent_content": node.metadata.get("parent_content", "") 
                }
                
                if cid not in chunk_results or result_dict["combined_score"] > chunk_results[cid].get("combined_score", 0):
                    chunk_results[cid] = result_dict

        all_results = []
        for cid, result in chunk_results.items():
            query_boost = 1.0 + (0.1 * (chunk_query_hits[cid] - 1))
            original_score = result.get("combined_score", 0)
            result["combined_score"] = original_score * query_boost
            result["query_hit_count"] = chunk_query_hits[cid]
            all_results.append(result)
            
            logger.debug(f"Node {cid[:8]} | Orig Score: {original_score:.3f} | Hits: {chunk_query_hits[cid]} | Boosted: {result['combined_score']:.3f}")
        
        all_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        if not all_results:
            logger.warning("No results found! Check if 'workspace_id' in the DB matches the current workspace.")
            
        validated, note = await ValidationService.validate_search_results(all_results, min_score=0.10)
        logger.info(f"Validation Note: {note} | Final validated results: {len(validated)}")
        
        return {"search_results": validated[:50]}

    async def rerank_results(self, state: RAGState) -> Dict:
        """IMPROVED: Better reranking with fixed weights"""
        docs = state.get("search_results") or []
        if not docs:
            return {"reranked_results": []}

        # Use cross-encoder for small sets, hybrid for larger
        method = "cross_encoder" if len(docs) <= 20 else "hybrid"

        reranked = await self.reranking_service.rerank_documents(
            query=state["query"],
            documents=docs,
            method=method,
            top_k=20  # Increased from 15
        )
        logger.info(f"Reranked {len(reranked)} results using method: {method}")
        return {"reranked_results": reranked}
    
    async def rank_results(self, state: RAGState) -> Dict:
        """Apply multi-factor ranking to search results"""
        logger.info(f"Ranking results")
        results_to_rank = state.get("reranked_results") or state["search_results"]
        ranked = await self.ranking_service.rank_results(
            results=results_to_rank,
            query=state["query"]
        )
        
        # Lower threshold from 0.3 to 0.2 to include more relevant docs
        top_results = [r for r in ranked if r["final_score"] > 0.2][:15]
        
        # Ensure we always have at least some results
        if not top_results and ranked:
            top_results = ranked[:5]
        
        logger.info(f"Selected {len(top_results)} top-ranked results")
        
        return {
            "ranked_results": top_results
        }
    
    async def assemble_context(self, state: RAGState) -> Dict:
        """
        FIXED: Use correct token budget from settings and handle single-source case
        """
        logger.info("Assembling context with token budget")

        ranked_results = state.get("ranked_results") or []
        if not ranked_results:
            return {"context": "No information found.", "context_sources": []}

        # Use the actual DEFAULT_TOKEN_BUDGET from settings
        token_budget = int(settings.DEFAULT_TOKEN_BUDGET)  # Should be 8000
        primary_budget = int(token_budget * settings.PRIMARY_SOURCES_RATIO)
        supporting_budget = int(token_budget * settings.SUPPORTING_CONTEXT_RATIO)
        
        logger.info(f"Token budget: {token_budget} (primary: {primary_budget}, supporting: {supporting_budget})")

        max_total_budget = primary_budget + supporting_budget

        # Adaptive parameters based on number of sources available
        num_sources = len(ranked_results)
        if num_sources == 1:
            # Single source - use full content
            GUARANTEED_TOPK = 1
            MAX_TOKENS_PER_CHUNK = 2000  # Much larger for single source
            GUARANTEED_MAX_TOKENS = 1500
            MIN_SOURCES_TARGET = 1
        elif num_sources <= 3:
            # Few sources - be generous
            GUARANTEED_TOPK = num_sources
            MAX_TOKENS_PER_CHUNK = 800
            GUARANTEED_MAX_TOKENS = 500
            MIN_SOURCES_TARGET = num_sources
        else:
            # Many sources - balance coverage
            GUARANTEED_TOPK = 5
            MAX_TOKENS_PER_CHUNK = 500
            GUARANTEED_MAX_TOKENS = 300
            MIN_SOURCES_TARGET = 5

        def approx_tokens(text: str) -> int:
            return int(len(text.split()) * 1.3)

        def truncate_to_tokens(text: str, max_tokens: int) -> str:
            if not text:
                return ""
            words = text.split()
            max_words = max(1, int(max_tokens / 1.3))
            if len(words) <= max_words:
                return text
            return " ".join(words[:max_words]) + " ..."

        context_parts: list[str] = []
        context_sources: list[dict] = []
        processed_ids: set[str] = set()

        total_tokens = 0
        source_index = 1

        def add_source(result: dict, text: str) -> bool:
            nonlocal total_tokens, source_index

            chunk_id = str(result.get("chunk_id", "unknown"))
            t = approx_tokens(text)

            if not text.strip():
                return False

            if total_tokens + t > max_total_budget:
                return False

            # Include relevance score and query hit count if available
            score = result.get("final_score", 0)
            hits = result.get("query_hit_count", 1)
            filename = result.get("filename", "unknown")
            
            context_parts.append(
                f"[Source {source_index}] (file: {filename}, relevance: {score:.2f}, found_by: {hits} queries)\n{text}\n"
            )
            context_sources.append(result)
            total_tokens += t
            processed_ids.add(chunk_id)
            source_index += 1
            return True
        
        # Phase 1: Guarantee top-K sources
        for result in ranked_results[:GUARANTEED_TOPK]:
            chunk_id = str(result.get("chunk_id", "unknown"))
            if chunk_id in processed_ids:
                continue

            content = result.get("parent_content") or result.get("content") or ""
            preview = truncate_to_tokens(content, GUARANTEED_MAX_TOKENS)
            added = add_source(result, preview)
            
            # Force add even if budget exceeded (but truncate more)
            if not added and total_tokens == 0:
                tiny_preview = truncate_to_tokens(content, 100)
                add_source(result, tiny_preview)

        # Phase 2: Add full chunks up to primary budget
        for result in ranked_results:
            if total_tokens >= primary_budget:
                break

            chunk_id = str(result.get("chunk_id", "unknown"))
            if chunk_id in processed_ids:
                continue

            content = result.get("parent_content") or result.get("content") or ""
            capped = truncate_to_tokens(content, MAX_TOKENS_PER_CHUNK)

            t = approx_tokens(capped)
            if total_tokens + t > primary_budget:
                # Try with smaller snippet
                smaller = truncate_to_tokens(content, MAX_TOKENS_PER_CHUNK // 2)
                t2 = approx_tokens(smaller)
                if total_tokens + t2 <= primary_budget:
                    add_source(result, smaller)
                continue

            add_source(result, capped)

        # Phase 3: Supporting context up to full budget
        for result in ranked_results:
            if total_tokens >= max_total_budget:
                break

            chunk_id = str(result.get("chunk_id", "unknown"))
            if chunk_id in processed_ids:
                continue

            content = result.get("parent_content") or result.get("content") or ""
            capped = truncate_to_tokens(content, MAX_TOKENS_PER_CHUNK // 2)

            add_source(result, capped)

        # Phase 4: Ensure minimum diversity
        if len(context_sources) < MIN_SOURCES_TARGET:
            for result in ranked_results:
                if len(context_sources) >= MIN_SOURCES_TARGET:
                    break

                chunk_id = str(result.get("chunk_id", "unknown"))
                if chunk_id in processed_ids:
                    continue

                content = result.get("parent_content") or result.get("content") or ""
                snippet = truncate_to_tokens(content, 150)
                add_source(result, snippet)

        context = "\n".join(context_parts).strip()

        logger.info(
            f"✓ Assembled context: sources={len(context_sources)}, tokens={total_tokens}/{max_total_budget}"
        )
        
        # Warning if using very little of budget
        usage_pct = (total_tokens / max_total_budget) * 100
        if usage_pct < 10 and num_sources > 1:
            logger.warning(f"⚠️ Only using {usage_pct:.1f}% of token budget - may indicate data issue")

        return {
            "context": context if context else "No information found.",
            "context_sources": context_sources,
        }

    async def generate_response(self, state: RAGState) -> Dict:
        """
        FIXED: Simplified prompts optimized for smaller LLMs like Gemma 4B
        """
        logger.info("Generating response")
        
        ranked = state.get("ranked_results") or []
        if not ranked:
            return {
                "response": "I could not find relevant information to answer your question.",
                "messages": [AIMessage(content="No relevant documents found.")]
            }
        
        max_score = max([r.get("final_score", 0) for r in ranked[:5]], default=0)
        if max_score < 0.15:
            return {
                "response": "The available documents don't contain sufficiently relevant information to answer this question accurately.",
                "messages": [AIMessage(content="Low relevance documents.")]
            }
        
        # Simpler, clearer prompt for small LLMs
        # Small LLMs work better with shorter, more direct instructions
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You answer questions using only the provided sources. 
            You must add [Source N] after every fact.
            If the answer is not in the sources, say "Not found in documents."
            """),
            ("user", """Sources:
            {context}

            Question: {query}

            Instructions:
            1. Read all sources
            2. Write the answer
            3. Add [Source N] after each fact
            4. Use facts from multiple sources if available

            Answer:""")
        ])
        
        messages = [
            SystemMessage(content=prompt.messages[0].prompt.template),
            HumanMessage(
                content=prompt.messages[1].prompt.template.format(
                    context=state["context"], query=state["query"]
                )
            ),
        ]
        
        try:
            text = await self.llm.generate(messages)
            response = LLMResponse(text)
            response_text = response.content
            
            # If LLM forgot citations, try to add them
            if '[Source' not in response_text and 'not found' not in response_text.lower():
                logger.warning("LLM response missing citations - attempting auto-fix")
                # Add a generic source citation if answer seems valid
                if len(response_text) > 20 and state.get("context_sources"):
                    response_text += " [Source 1]"
            
            logger.info("Response generated successfully")
            
            return {
                "response": response_text,
                "messages": [AIMessage(content=response_text)]
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "response": "I apologize, but I encountered an error generating a response.",
                "messages": [AIMessage(content="Error occurred")]
            }
    
    async def verify_citations(self, state: RAGState) -> Dict:
        """Verify that citations in the response are grounded in sources"""
        logger.info("Verifying citations")
        
        verification_results = await self.citation_service.verify_citations(
            response=state["response"],
            sources=state.get("context_sources", [])
        )
        
        # Include more source metadata
        citations = []
        for i, result in enumerate(state["context_sources"][:10], 1):
            citations.append({
                "source_number": i,
                "chunk_id": result["chunk_id"],
                "resource_id": result["resource_id"],
                "filename": result.get("filename", "unknown"),
                "content_preview": result["content"][:250] + "...",
                "relevance_score": result.get("final_score", result.get("combined_score", 0))
            })
        
        metadata = {
            "num_sources_used": len(state.get("context_sources", [])),
            "num_sources_cited": len(citations),
            "verification_passed": verification_results.get("passed", True),
            "verification_issues": verification_results.get("issues", [])
        }
        
        return {
            "citations": citations,
            "metadata": metadata
        }
        
    async def self_reflect(self, state: RAGState) -> Dict:
        """IMPROVED: Check if answer quality meets standards"""
        try:
            reflection = await self.query_decomposer.self_reflect(
                state["query"], 
                state["response"], 
                state["ranked_results"]
            )
            # Lower threshold from 7 to 6 to reduce unnecessary refinements
            needs_refinement = reflection["score"] < 6 and state.get("refinement_count", 0) < 1
            logger.info(f"Self-reflection score: {reflection['score']}/10, needs_refinement: {needs_refinement}")
            return {"self_reflection": reflection, "needs_refinement": needs_refinement}
        except Exception as e:
            logger.warning(f"Self-reflection failed: {e}")
            return {"needs_refinement": False}

    async def refine_response(self, state: RAGState) -> Dict:
        """
        IMPROVED: Simpler refinement for small LLMs
        """
        logger.info("Refining response based on self-reflection...")
        
        # For very short responses, generate a new one instead of refining
        if len(state["response"]) < 30:
            logger.info("Response too short, regenerating instead of refining")
            return await self.generate_response(state)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Fix this answer to include proper citations."),
            ("user", """Sources:
            {context}

            Current Answer:
            {response}

            Task: Add [Source N] citations to every fact. Remove any unsupported claims.

            Fixed Answer:""")
        ])
        
        messages = [
            SystemMessage(content=prompt.messages[0].prompt.template),
            HumanMessage(
                content=prompt.messages[1].prompt.template.format(
                    context=state["context"], response=state["response"]
                )
            ),
        ]
        
        try:
            text = await self.llm.generate(messages)
            resp = LLMResponse(text)
            refined = resp.content
            
            # If refinement made it worse (removed content), keep original
            if len(refined) < len(state["response"]) * 0.5:
                logger.warning("Refinement removed too much content, keeping original")
                refined = state["response"]
            
            return {
                "response": refined,
                "refinement_count": state.get("refinement_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return {
                "response": state["response"],  # Keep original
                "refinement_count": state.get("refinement_count", 0) + 1
            }
    
    async def run(self, query: str, workspace_id: UUID, conversation_id: UUID) -> Dict:
        """Run the RAG pipeline"""
        logger.info(f"Running RAG pipeline for query: {query[:100]}...")
        
        initial_state = {
            "query": query,
            "workspace_id": workspace_id,
            "conversation_id": conversation_id,
            "expanded_queries": [],
            "search_results": [],
            "reranked_results": [],
            "ranked_results": [],
            "context": "",
            "response": "",
            "citations": [],
            "metadata": {},
            "context_sources": [],
            "self_reflection": {},
            "needs_refinement": False,
            "refinement_count": 0,
            "messages": []
        }
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            return final_state
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            raise
        
    async def run_stream(
        self,
        query: str,
        workspace_id: UUID,
        conversation_id: UUID
    ) -> AsyncGenerator[dict, None]:
        """
        IMPROVED: Stream with better status messages
        """
        state: RAGState = {
            "query": query,
            "workspace_id": workspace_id,
            "conversation_id": conversation_id,
            "expanded_queries": [],
            "search_results": [],
            "reranked_results": [],
            "ranked_results": [],
            "context": "",
            "response": "",
            "citations": [],
            "metadata": {},
            "context_sources": [],
            "self_reflection": {},
            "needs_refinement": False,
            "refinement_count": 0,
            "messages": []
        }

        try:
            yield {"type": "status", "data": "Analyzing question..."}
            out = await self.expand_query(state)
            state.update(out)

            yield {"type": "status", "data": f"Searching across {len(state['expanded_queries'])} query variants..."}
            #out = await self.hybrid_search(state)
            out = await self.retrieve_data(state)  # Use LlamaIndex retrieval
            state.update(out)

            yield {"type": "status", "data": f"Reranking {len(state.get('search_results', []))} results..."}
            out = await self.rerank_results(state)
            state.update(out)

            yield {"type": "status", "data": "Applying multi-factor ranking..."}
            out = await self.rank_results(state)
            state.update(out)

            yield {"type": "status", "data": f"Assembling context from {len(state.get('ranked_results', []))} sources..."}
            out = await self.assemble_context(state)
            state.update(out)

            yield {"type": "status", "data": "Generating answer..."}
            
            # IMPROVED: Better streaming prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a precise research assistant. Your answers must:
                1. Use ONLY facts from provided sources
                2. Cite every claim with [Source N]
                3. Synthesize multiple sources when available
                4. Be comprehensive and accurate"""),
                                ("user", """Answer using the provided sources. Follow these steps mentally:

                1. Scan all [Source] blocks
                2. Identify relevant facts from each source
                3. Synthesize into a coherent answer
                4. Cite every fact

                Context:
                {context}

                Question: {query}

                Answer (start directly, include citations):""")
            ])

            messages = [
                SystemMessage(content=prompt.messages[0].prompt.template),
                HumanMessage(
                    content=prompt.messages[1].prompt.template.format(
                        context=state["context"], query=state["query"]
                    )
                ),
            ]
            
            full_text = ""
            
            async for token in self.llm.stream(messages):
                full_text += token
                yield {"type": "content", "data": token}

            state["response"] = full_text

            yield {"type": "status", "data": "Verifying citations..."}
            out = await self.verify_citations(state)
            state.update(out)

            yield {"type": "status", "data": "Performing quality check..."}
            out = await self.self_reflect(state)
            state.update(out)

            if state.get("needs_refinement") and state.get("refinement_count", 0) < 1:
                yield {"type": "status", "data": "Refining answer..."}
                out = await self.refine_response(state)
                state.update(out)
                out = await self.verify_citations(state)
                state.update(out)

            yield {
                "type": "end",
                "response": state["response"],
                "citations": state.get("citations", []),
                "metadata": state.get("metadata", {})
            }

        except Exception as e:
            logger.exception("run_stream failed")
            yield {"type": "error", "data": str(e)}