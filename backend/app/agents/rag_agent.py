"""
LangGraph-based RAG agent for orchestrating the retrieval pipeline
"""
import logging
from typing import TypedDict, List, Dict, Any, Annotated, AsyncGenerator
from uuid import UUID
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.services.hardware import HardwareDetector
from app.agents.prompts import RAG_SYSTEM_PROMPT, QUERY_EXPANSION_PROMPT
from app.services.enhanced_rag import AdvancedRerankingService, QueryDecompositionAgent, ValidationService, retry_with_backoff, CircuitBreaker

logger = logging.getLogger(__name__)


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
    
    def __init__(self, search_service, ranking_service, citation_service):
        self.search_service = search_service
        self.ranking_service = ranking_service
        self.citation_service = citation_service
        
        # Initialize enhanced services
        self.reranking_service = AdvancedRerankingService(search_service.db)
        self.validation_service = ValidationService()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize query decomposition
        self.query_decomposer = QueryDecompositionAgent(self.llm)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _initialize_llm(self):
        """Initialize LLM based on hardware and configuration"""
        # Check if Gemini API key is available
        if settings.GEMINI_API_KEY:
            logger.info("Using Google Gemini")
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.3
            )
        
        # Use Ollama with hardware-optimized model
        optimal_model = HardwareDetector.get_optimal_model()
        ollama_options = HardwareDetector.get_ollama_options()
        
        logger.info(f"Using Ollama model: {optimal_model}")
        
        return ChatOllama(
            model=optimal_model,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=ollama_options["temperature"],
            top_p=ollama_options["top_p"],
            num_predict=ollama_options["num_predict"]
        )
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("expand_query", self.expand_query)
        workflow.add_node("hybrid_search", self.hybrid_search)
        workflow.add_node("rerank_results", self.rerank_results)
        workflow.add_node("rank_results", self.rank_results)
        workflow.add_node("assemble_context", self.assemble_context)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("verify_citations", self.verify_citations)
        workflow.add_node("self_reflect", self.self_reflect)
        workflow.add_node("refine_response", self.refine_response)
        
        # Define edges
        workflow.set_entry_point("expand_query")
        workflow.add_edge("expand_query", "hybrid_search")
        workflow.add_edge("hybrid_search", "rerank_results")
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
        workflow.add_edge("verify_citations", END)
        
        return workflow.compile()
    
    async def expand_query(self, state: RAGState) -> Dict:
        """Expand the user query into multiple variants"""
        logger.info(f"Expanding query: {state['query']}")
        
        try:
            sub_queries = await self.query_decomposer.decompose_query(state["query"])
            # always include original query first
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
        """Perform hybrid search across all query variants"""
        logger.info(f"Performing hybrid search for {len(state['expanded_queries'])} queries")
        
        all_results = []
        seen_chunk_ids = set()
        
        async def run_one(q: str):
            return await self.search_service.hybrid_search(
                query=q,
                workspace_id=state["workspace_id"],
                limit=settings.MAX_SEARCH_RESULTS
            )

        for q in state["expanded_queries"]:
            try:
                results = await retry_with_backoff(lambda: run_one(q))
            except Exception as e:
                logger.warning(f"Hybrid search failed for query='{q}': {e}")
                continue

            for r in results:
                cid = str(r["chunk_id"])
                if cid not in seen_chunk_ids:
                    all_results.append(r)
                    seen_chunk_ids.add(cid)

        validated, note = await ValidationService.validate_search_results(all_results)
        if note:
            logger.warning(f"Search validation note: {note}")
        
        logger.info(f"Found {len(validated)} unique search results")
        
        return {
            "search_results": validated[:50]  # Limit to top 80
        }

    async def rerank_results(self, state: RAGState) -> Dict:
        docs = state.get("search_results") or []
        if not docs:
            return {"reranked_results": []}

        method = "cross_encoder" if len(docs) <= 25 else "hybrid"

        reranked = await self.reranking_service.rerank_documents(
            query=state["query"],
            documents=docs,
            method=method,
            top_k=15
        )
        logger.info(f"Reranked results using method: {method}")
        logger.info(f"len reranked: {len(reranked)}")
        return {"reranked_results": reranked}
    
    async def rank_results(self, state: RAGState) -> Dict:
        """Apply multi-factor ranking to search results"""
        logger.info(f"Ranking {len(state['search_results'])} results")
        results_to_rank = state.get("reranked_results") or state["search_results"]
        ranked = await self.ranking_service.rank_results(
            results=results_to_rank,
            query=state["query"]
        )
        
        # Select top results based on score threshold
        top_results = [r for r in ranked if r["final_score"] > 0.3][:10]
        
        logger.info(f"Selected {len(top_results)} top-ranked results")
        
        return {
            "ranked_results": top_results
        }
    
    async def assemble_context(self, state: RAGState) -> Dict:
        """Assemble context from ranked results with token budgeting"""
        logger.info("Assembling context with token budget")
        
        token_budget = settings.DEFAULT_TOKEN_BUDGET
        primary_budget = int(token_budget * settings.PRIMARY_SOURCES_RATIO)
        supporting_budget = int(token_budget * settings.SUPPORTING_CONTEXT_RATIO)
        
        if not state["ranked_results"]:
            return {"context": "No information found."}
        
        context_parts = []
        total_tokens = 0
        context_sources = []
        
        # Add primary sources (top-ranked)
        primary_sources = state["ranked_results"][:5]
        for i, result in enumerate(primary_sources, 1):
            chunk_text = result["content"]
            chunk_tokens = len(chunk_text.split()) * 1.3  # Approximate tokens
            
            if total_tokens + chunk_tokens <= primary_budget:
                context_sources.append(result)
                context_parts.append(f"[Source {i}]\n{chunk_text}\n")
                total_tokens += chunk_tokens
        
        # Add supporting context if budget allows
        supporting_sources = state["ranked_results"][5:10]
        for i, result in enumerate(supporting_sources, len(primary_sources) + 1):
            chunk_text = result["content"]
            chunk_tokens = len(chunk_text.split()) * 1.3
            
            if total_tokens + chunk_tokens <= supporting_budget:
                context_sources.append(result)
                context_parts.append(f"[Source {i}]\n{chunk_text}\n")
                total_tokens += chunk_tokens
        
        context = "\n".join(context_parts)
        
        logger.info(f"Assembled context with ~{int(total_tokens)} tokens")
        
        return {
            "context": context,
            "context_sources": context_sources
        }
    
    async def generate_response(self, state: RAGState) -> Dict:
        """Generate response using LLM with assembled context"""
        logger.info("Generating response")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("user", """
            You MUST follow these rules:
            1) Use ONLY facts from the Context.
            2) Every factual sentence must end with a citation like [Source 1].
            3) If the answer is not in Context at all, reply ONLY with: "Not found in the provided documents."
            4) Do NOT add "Not found..." if you already answered using Context.
            
            Here is the information you have:
            Context: {context}

            Question: {query}

            Provide a detailed answer with citations.
            """)])
        
        chain = prompt | self.llm
        
        try:
            response = await chain.ainvoke({
                "context": state["context"],
                "query": state["query"]
            })
            
            logger.info("Response generated successfully")
            
            return {
                "response": response.content,
                "messages": [AIMessage(content=response.content)]
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
        
        citations = []
        for result in state["context_sources"][:5]:  # Top 5 sources
            citations.append({
                "chunk_id": result["chunk_id"],
                "resource_id": result["resource_id"],
                "content_preview": result["content"][:200] + "...",
                "relevance_score": result["final_score"]
            })
        
        metadata = {
            "num_sources_used": len(state["ranked_results"]),
            "verification_passed": verification_results.get("passed", True),
            "verification_issues": verification_results.get("issues", [])
        }
        
        return {
            "citations": citations,
            "metadata": metadata
        }
        
    async def self_reflect(self, state: RAGState) -> Dict:
        """Check if answer is good"""
        try:
            reflection = await self.query_decomposer.self_reflect(
                state["query"], state["response"], state["ranked_results"]
            )
            needs_refinement = reflection["score"] < 7 and state.get("refinement_count", 0) < 1
            return {"self_reflection": reflection, "needs_refinement": needs_refinement}
        except:
            return {"needs_refinement": False}

    async def refine_response(self, state: RAGState) -> Dict:
        """Refine if needed"""
        logger.info("Refining response based on self-reflection...")
        prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict citation enforcer."),
        ("user", """Fix the answer using ONLY the Context.
                    Rules:
                    - Remove any claim not supported by Context.
                    - Every sentence MUST include [Source N].
                    - If not enough info, output exactly: Not found in the provided documents.

                    Context:
                    {context}

                    Original Answer:
                    {answer}

                    Return the corrected answer:""")])
        
        chain = prompt | self.llm
        resp = await chain.ainvoke({"context": state["context"], "answer": state["response"]})
        return {"response": resp.content, "refinement_count": state.get("refinement_count", 0) + 1}
    
    
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
        Stream tokens directly from the LLM while still running full RAG pipeline.
        Yields events: status/content/end/error
        """

        # Build initial state
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
            yield {"type": "status", "data": "Expanding query..."}
            out = await self.expand_query(state)
            state.update(out)

            yield {"type": "status", "data": "Searching documents..."}
            out = await self.hybrid_search(state)
            state.update(out)

            yield {"type": "status", "data": "Reranking results..."}
            out = await self.rerank_results(state)
            state.update(out)

            yield {"type": "status", "data": "Ranking results..."}
            out = await self.rank_results(state)
            state.update(out)

            yield {"type": "status", "data": "Assembling context..."}
            out = await self.assemble_context(state)
            state.update(out)

            # --- REAL TOKEN STREAMING STARTS HERE ---
            yield {"type": "status", "data": "Generating answer..."}
            prompt = ChatPromptTemplate.from_messages([
                ("system", RAG_SYSTEM_PROMPT),
                ("user", """
                You MUST follow these rules:
                1) Use ONLY facts from the Context.
                2) Every factual sentence must end with a citation like [Source 1].
                3) If the answer is not in Context at all, reply ONLY with: "Not found in the provided documents."
                4) Do NOT add "Not found..." if you already answered using Context.

                Here is the information you have:
                Context: {context}

                Question: {query}

                Provide a detailed answer with citations.
                """)
            ])

            chain = prompt | self.llm  

            full_text = ""
            async for chunk in chain.astream({
                "context": state["context"],
                "query": state["query"]
            }):
                token = getattr(chunk, "content", "") or ""
                if token:
                    full_text += token
                    yield {"type": "content", "data": token}

            state["response"] = full_text

            yield {"type": "status", "data": "Verifying citations..."}
            out = await self.verify_citations(state)
            state.update(out)

            yield {"type": "status", "data": "Finalizing..."}
            out = await self.self_reflect(state)
            state.update(out)

            if state.get("needs_refinement") and state.get("refinement_count", 0) < 1:
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