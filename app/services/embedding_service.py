"""
Embedding Service
Handles generation of embeddings using OpenAI's API.
"""

from typing import List
from openai import AsyncOpenAI
from app.config import settings


class EmbeddingService:
    """Service for generating text embeddings using OpenAI."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize the embedding service.

        Args:
            api_key: OpenAI API key (optional, uses settings if not provided)
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file.")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-small"  # 1536 dimensions
        self.dimensions = 1536

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        Processes in batches for efficiency.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)

        Raises:
            Exception: If embedding generation fails
        """
        if not texts:
            return []

        try:
            # OpenAI API handles batching internally
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )

            # Extract embeddings in the same order as input
            embeddings = [item.embedding for item in response.data]

            return embeddings

        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")

    async def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector (list of floats)
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.

        Returns:
            int: Embedding dimension (1536 for text-embedding-3-small)
        """
        return self.dimensions
