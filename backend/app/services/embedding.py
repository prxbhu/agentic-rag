"""
Embedding service for generating vector embeddings
"""
import logging
from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.services.hardware import HardwareDetector

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers"""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to reuse model across requests"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize embedding model"""
        if self._model is None:
            self._initialize_model()
    
    def _initialize_model(self):
        """Load the embedding model with hardware optimization"""
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        
        device = HardwareDetector.get_embedding_device()
        logger.info(f"Using device: {device}")
        
        try:
            self._model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=device
            )
            
            # Verify model dimension
            test_embedding = self._model.encode(["test"], show_progress_bar=False)
            actual_dim = test_embedding.shape[1]
            
            if actual_dim != settings.EMBEDDING_DIMENSION:
                logger.warning(
                    f"Model dimension {actual_dim} doesn't match config "
                    f"{settings.EMBEDDING_DIMENSION}. Updating config."
                )
                # In production, you'd want to update the database schema
            
            logger.info(
                f"Model loaded successfully. Dimension: {actual_dim}, Device: {device}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            embedding = self._model.encode(
                [text],
                show_progress_bar=False,
                convert_to_numpy=True
            )[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing (uses hardware-optimized default if None)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if batch_size is None:
            batch_size = HardwareDetector.get_batch_size()
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts with batch_size={batch_size}")
            
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True
            )
            
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        return float(similarity)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": settings.EMBEDDING_MODEL,
            "dimension": settings.EMBEDDING_DIMENSION,
            "device": HardwareDetector.get_embedding_device(),
            "max_sequence_length": self._model.max_seq_length if self._model else None
        }


# Global instance
embedding_service = EmbeddingService()