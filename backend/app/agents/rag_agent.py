"""
LangGraph-based RAG agent for orchestrating the retrieval pipeline
"""
import logging
from typing import TypedDict, List, Dict, Any, Annotated
from uuid import UUID
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

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
        
        workflow.add_edge("refine_response", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    async def expand_query(self, state: RAGState) -> Dict:
        """Expand the user query into multiple variants"""
        logger.info(f"Expanding query: {state['query']}")
        
        if len(state["query"].split()) < 5:
                return {"expanded_queries": [state["query"]], "messages": []}
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_EXPANSION_PROMPT),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm
        
        try:
            response = await chain.ainvoke({"query": state["query"]})
            variants = [v.strip() for v in response.content.strip().split("\n") if v.strip()]
            queries = [state["query"]] + variants[:3]
            return {"expanded_queries": queries, "messages": [HumanMessage(content=state["query"])]}
            
            # # Parse expanded queries (expecting line-separated variants)
            # expanded = response.content.strip().split("\n")
            # expanded_queries = [q.strip() for q in expanded if q.strip()]
            
            # # Ensure original query is included
            # if state["query"] not in expanded_queries:
            #     expanded_queries.insert(0, state["query"])
            
            # logger.info(f"Generated {len(expanded_queries)} query variants")
            
            # return {
            #     "expanded_queries": expanded_queries[:5],  # Limit to 5 variants
            #     "messages": [HumanMessage(content=state["query"])]
            # }
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
        
        for query in state["expanded_queries"]:
            results = await self.search_service.hybrid_search(
                query=query,
                workspace_id=state["workspace_id"],
                limit=settings.MAX_SEARCH_RESULTS
            )
            
            # Deduplicate while preserving order
            for result in results:
                if result["chunk_id"] not in seen_chunk_ids:
                    all_results.append(result)
                    seen_chunk_ids.add(result["chunk_id"])
        
        logger.info(f"Found {len(all_results)} unique search results")
        
        return {
            "search_results": all_results[:50]  # Limit to top 50
        }

    async def rerank_results(self, state: RAGState) -> Dict:
        if not state["search_results"]:
            return {"reranked_results": []}
        
        # Use 'cross_encoder' if available (most accurate), else 'hybrid' (MMR + Semantic)
        # For this specific case (Syllabus), Cross Encoder is best at matching "GATE 2026" specifically.
        reranked = await self.reranking_service.rerank_documents(
            query=state["query"],
            documents=state["search_results"],
            method="cross_encoder", # Safer default "hybrid", falls back gracefully
            top_k=15
        )
        
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
        
        # Add primary sources (top-ranked)
        primary_sources = state["ranked_results"][:5]
        for i, result in enumerate(primary_sources, 1):
            chunk_text = result["content"]
            chunk_tokens = len(chunk_text.split()) * 1.3  # Approximate tokens
            
            if total_tokens + chunk_tokens <= primary_budget:
                context_parts.append(f"[Source {i}]\n{chunk_text}\n")
                total_tokens += chunk_tokens
        
        # Add supporting context if budget allows
        supporting_sources = state["ranked_results"][5:10]
        for i, result in enumerate(supporting_sources, len(primary_sources) + 1):
            chunk_text = result["content"]
            chunk_tokens = len(chunk_text.split()) * 1.3
            
            if total_tokens + chunk_tokens <= supporting_budget:
                context_parts.append(f"[Source {i}]\n{chunk_text}\n")
                total_tokens += chunk_tokens
        
        context = "\n".join(context_parts)
        
        logger.info(f"Assembled context with ~{int(total_tokens)} tokens")
        
        return {
            "context": context
        }
    
    async def generate_response(self, state: RAGState) -> Dict:
        """Generate response using LLM with assembled context"""
        logger.info("Generating response")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("user", """Context:
            {context}

            Question: {query}

            Provide a comprehensive answer using ONLY the information from the provided context. 
            Always cite sources using [Source N] format. If information is not in the context, 
            state that it's not available.""")
                    ])
        
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
            sources=state["ranked_results"]
        )
        
        citations = []
        for result in state["ranked_results"][:5]:  # Top 5 sources
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
        return {"refinement_count": state.get("refinement_count", 0) + 1}
    
    
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