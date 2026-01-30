"""
LLM service abstraction for multiple providers
"""
import logging
from typing import Optional, AsyncGenerator
from abc import ABC, abstractmethod
import os
from fastapi import HTTPException
import httpx
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI

from app.config import settings
from dotenv import load_dotenv
from google.oauth2 import service_account

load_dotenv()


logger = logging.getLogger(__name__)

class LLMResponse:
    """Ensures response.content exists for downstream logic"""
    def __init__(self, content: str):
        self.content = content

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]  
credentials_path = PROJECT_ROOT / "vertex.json"
credentials = service_account.Credentials.from_service_account_file(str(credentials_path), scopes=["https://www.googleapis.com/auth/cloud-platform"])
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

class GeminiLLMService(BaseLLMService):
    """Google Gemini LLM service implementation"""
    
    def __init__(self):
        logger.info("Initializing Google Gemini")
        
        self.llm = ChatVertexAI(
            model="gemini-2.5-pro",
            credentials=credentials,
            temperature=0.0,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION")
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

class vllmGemmaService(BaseLLMService):
    """vLLM Gemma LLM service implementation"""
    
    def __init__(self):
        logger.info("Initializing vLLM Gemma")
        self.base_url = os.environ.get("VLLM_BASE_URL") 
        if not self.base_url:
            raise RuntimeError("VLLM_BASE_URL is not configured")
        self.model = "gemma-3-4b-it"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=300.0,
            headers={"Content-Type": "application/json"}
        )

        logger.info("Initialized vLLM Gemma service")
        
    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict]:
        """Convert LangChain messages to OpenAI format"""
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                continue

            formatted.append({"role": role, "content": msg.content})

        return formatted
    
    async def generate(
        self,
        messages: list[BaseMessage],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response from vLLM Gemma
        """
        try:
            payload = {
                "model": self.model,
                "messages": self._convert_messages(messages),
                "temperature": temperature,
                "max_tokens": max_tokens or 1024,
            }
            response = await self.client.post("/v1/chat/completions",json=payload,)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error(f"vLLM Gemma generation failed: {e}")
            raise HTTPException(status_code=500, detail="LLM generation failed")
    
    async def stream(
        self,
        messages: list[BaseMessage],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from vLLM Gemma
        """
        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens or 1024,
            "stream": True,
        }
        try:
            async with self.client.stream("POST", "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.replace("data:", "").strip()
                    if data == "[DONE]":
                        break
                    chunk = httpx.Response(status_code=200, content=data).json()
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]
        except Exception as e:
            logger.error(f"vLLM Gemma streaming failed: {e}")
            raise HTTPException(status_code=500, detail="LLM streaming failed")

class LLMServiceFactory:
    """Factory for creating LLM service instances"""
    
    @staticmethod
    def create_llm_service(
        provider: Optional[str] = None,
    ) -> BaseLLMService:
        """
        Create an LLM service instance
        
        Args:
            provider: LLM provider ('ollama', 'gemini', or None for auto)
            api_key: API key for cloud providers
            
        Returns:
            LLM service instance
        """
        logger.info(f"Auto-detect: GEMINI={bool(credentials_path.exists())}, VLLM={bool(os.environ.get('VLLM_BASE_URL'))}")
        # Auto-detect provider
        if provider is None:
            if credentials_path.exists():
                provider = "gemini"
            elif os.environ.get("VLLM_BASE_URL"):
                provider = "vllm"
        logger.info(f"Creating LLM service: {provider}")
        
        if provider == "gemini":
            return GeminiLLMService()
        elif provider == "vllm":
            return vllmGemmaService()
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