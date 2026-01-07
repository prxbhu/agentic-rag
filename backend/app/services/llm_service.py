"""
LLM service abstraction for multiple providers
"""
import logging
from typing import Optional, AsyncGenerator
from abc import ABC, abstractmethod

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from app.config import settings
from app.services.hardware import HardwareDetector

logger = logging.getLogger(__name__)


class BaseLLMService(ABC):
    """Abstract base class for LLM services"""
    
    @abstractmethod
    async def generate(
        self,
        messages: list[BaseMessage],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a response"""
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: list[BaseMessage],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Stream a response"""
        pass


class OllamaLLMService(BaseLLMService):
    """Ollama LLM service implementation"""
    
    def __init__(self):
        # Get hardware-optimized model
        self.model_name = HardwareDetector.get_optimal_model()
        self.ollama_options = HardwareDetector.get_ollama_options()
        
        logger.info(f"Initializing Ollama with model: {self.model_name}")
        
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=self.ollama_options["temperature"],
            top_p=self.ollama_options["top_p"],
            num_predict=self.ollama_options["num_predict"]
        )
    
    async def generate(
        self,
        messages: list[BaseMessage],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response from Ollama
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            # Update temperature if different
            if temperature != self.ollama_options["temperature"]:
                self.llm.temperature = temperature
            
            # Generate response
            response = await self.llm.ainvoke(messages)
            
            return response.content
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    async def stream(
        self,
        messages: list[BaseMessage],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from Ollama
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Response chunks
        """
        try:
            # Update temperature if different
            if temperature != self.ollama_options["temperature"]:
                self.llm.temperature = temperature
            
            # Stream response
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise


class GeminiLLMService(BaseLLMService):
    """Google Gemini LLM service implementation"""
    
    def __init__(self, api_key: str):
        logger.info("Initializing Google Gemini")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )
    
    async def generate(
        self,
        messages: list[BaseMessage],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response from Gemini
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            # Update temperature
            self.llm.temperature = temperature
            
            # Generate response
            response = await self.llm.ainvoke(messages)
            
            return response.content
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
    
    async def stream(
        self,
        messages: list[BaseMessage],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from Gemini
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Response chunks
        """
        try:
            # Update temperature
            self.llm.temperature = temperature
            
            # Stream response
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content
        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            raise


class LLMServiceFactory:
    """Factory for creating LLM service instances"""
    
    @staticmethod
    def create_llm_service(
        provider: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> BaseLLMService:
        """
        Create an LLM service instance
        
        Args:
            provider: LLM provider ('ollama', 'gemini', or None for auto)
            api_key: API key for cloud providers
            
        Returns:
            LLM service instance
        """
        # Auto-detect provider
        if provider is None:
            if settings.GEMINI_API_KEY:
                provider = "gemini"
            else:
                provider = "ollama"
        
        logger.info(f"Creating LLM service: {provider}")
        
        if provider == "gemini":
            api_key = api_key or settings.GEMINI_API_KEY
            if not api_key:
                raise ValueError("Gemini API key is required")
            return GeminiLLMService(api_key)
        elif provider == "ollama":
            return OllamaLLMService()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# Global LLM service instance
_llm_service: Optional[BaseLLMService] = None


def get_llm_service() -> BaseLLMService:
    """
    Get the global LLM service instance (singleton)
    
    Returns:
        LLM service instance
    """
    global _llm_service
    
    if _llm_service is None:
        _llm_service = LLMServiceFactory.create_llm_service()
    
    return _llm_service


def reset_llm_service():
    """Reset the global LLM service instance"""
    global _llm_service
    _llm_service = None