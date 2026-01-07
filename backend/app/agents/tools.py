"""
LangChain tools for RAG agent
"""
import logging
from typing import Optional, List
from uuid import UUID

from langchain_core.tools import Tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    """Input schema for search tool"""
    query: str = Field(description="Search query to find relevant documents")
    workspace_id: str = Field(description="Workspace ID to search in")
    limit: int = Field(default=10, description="Maximum number of results")


class DocumentSearchTool:
    """Tool for searching documents in the knowledge base"""
    
    def __init__(self, search_service):
        self.search_service = search_service
    
    async def search(
        self,
        query: str,
        workspace_id: str,
        limit: int = 10
    ) -> List[dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            workspace_id: Workspace to search in
            limit: Maximum results
            
        Returns:
            List of relevant document chunks
        """
        try:
            results = await self.search_service.hybrid_search(
                query=query,
                workspace_id=UUID(workspace_id),
                limit=limit
            )
            
            # Format results for LLM
            formatted = []
            for result in results:
                formatted.append({
                    "content": result["content"],
                    "source": result.get("filename", "Unknown"),
                    "relevance": result.get("combined_score", 0)
                })
            
            return formatted
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def as_langchain_tool(self) -> Tool:
        """Convert to LangChain Tool"""
        return Tool(
            name="document_search",
            description="Search for relevant information in the knowledge base. "
                       "Use this when you need to find specific information to answer questions.",
            func=lambda query: self.search(query, "default", 10)
        )


class RetrievalInput(BaseModel):
    """Input schema for retrieval tool"""
    query: str = Field(description="Query to retrieve context for")
    num_results: int = Field(default=5, description="Number of results to retrieve")


class ContextRetrievalTool:
    """Tool for retrieving and assembling context"""
    
    def __init__(self, search_service, ranking_service):
        self.search_service = search_service
        self.ranking_service = ranking_service
    
    async def retrieve(
        self,
        query: str,
        workspace_id: str,
        num_results: int = 5
    ) -> str:
        """
        Retrieve and assemble context for a query
        
        Args:
            query: User query
            workspace_id: Workspace ID
            num_results: Number of results to retrieve
            
        Returns:
            Assembled context string
        """
        try:
            # Search
            results = await self.search_service.hybrid_search(
                query=query,
                workspace_id=UUID(workspace_id),
                limit=num_results * 2
            )
            
            # Rank
            ranked = await self.ranking_service.rank_results(
                results=results,
                query=query
            )
            
            # Assemble context
            context_parts = []
            for i, result in enumerate(ranked[:num_results], 1):
                context_parts.append(
                    f"[Source {i}]\n{result['content']}\n"
                )
            
            return "\n".join(context_parts)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return "No context available"
    
    def as_langchain_tool(self) -> Tool:
        """Convert to LangChain Tool"""
        return Tool(
            name="context_retrieval",
            description="Retrieve relevant context from the knowledge base to answer a question. "
                       "Returns formatted context with source citations.",
            func=lambda query: self.retrieve(query, "default", 5)
        )


def create_rag_tools(search_service, ranking_service=None) -> List[Tool]:
    """
    Create a list of RAG tools for LangChain agent
    
    Args:
        search_service: SearchService instance
        ranking_service: RankingService instance (optional)
        
    Returns:
        List of LangChain tools
    """
    tools = []
    
    # Add search tool
    search_tool = DocumentSearchTool(search_service)
    tools.append(search_tool.as_langchain_tool())
    
    # Add retrieval tool if ranking service is available
    if ranking_service:
        retrieval_tool = ContextRetrievalTool(search_service, ranking_service)
        tools.append(retrieval_tool.as_langchain_tool())
    
    return tools