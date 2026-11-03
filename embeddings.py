import logging
import numpy as np
from typing import List
from app.config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    @staticmethod
    async def get_embedding(text: str) -> List[float]:
        """Generate vector embedding for a given text using Gemini or deterministic fallback."""
        if settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                model_name = settings.EMBEDDING_MODEL or "models/text-embedding-004"
                result = genai.embed_content(
                    model=model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                if result and "embedding" in result:
                    return result["embedding"]
            except Exception as e:
                logger.warning(f"Gemini embedding API failed: {e}. Using vector fallback.")

        # Deterministic lightweight pseudo-embedding vector for offline / testing mode
        return EmbeddingService._fallback_vector(text)

    @staticmethod
    async def get_query_embedding(text: str) -> List[float]:
        """Generate vector embedding for a search query."""
        if settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                model_name = settings.EMBEDDING_MODEL or "models/text-embedding-004"
                result = genai.embed_content(
                    model=model_name,
                    content=text,
                    task_type="retrieval_query"
                )
                if result and "embedding" in result:
                    return result["embedding"]
            except Exception as e:
                logger.warning(f"Gemini query embedding API failed: {e}. Using vector fallback.")

        return EmbeddingService._fallback_vector(text)

    @staticmethod
    def _fallback_vector(text: str, dim: int = 128) -> List[float]:
        """Produce a normalized float vector based on character hash values for semantic simulation."""
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec.tolist()
        
        for w in words:
            h = hash(w) % dim
            vec[h] += 1.0
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vector lists."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
