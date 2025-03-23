import logging

from app.core.config import settings
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import Field


class JinaEmbeddings(HuggingFaceEmbeddings):
    task: str = Field(default="retrieval.passage")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_kwargs = {"device": "cuda", "trust_remote_code": True}
        self.encode_kwargs = {"normalize_embeddings": True}

    def set_task(self, task: str):
        """
        Set the task for the Jina embeddings.

        retrieval.query: Used for query embeddings in asymmetric retrieval tasks
        retrieval.passage: Used for passage embeddings in asymmetric retrieval tasks
        separation: Used for embeddings in clustering and re-ranking applications
        classification: Used for embeddings in classification tasks
        text-matching: Used for embeddings in tasks that quantify similarity between two texts, such as STS or symmetric retrieval tasks
        """
        self.task = task
    

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        import sentence_transformers  # type: ignore[import]

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        logging.info(f"Embedding with task:, {self.task}")
        if self.multi_process:
            pool = self._client.start_multi_process_pool()
            embeddings = self._client.encode_multi_process(texts, pool, task=self.task)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self._client.encode(
                texts,
                show_progress_bar=self.show_progress,
                task=self.task,
                **self.encode_kwargs,  # type: ignore
            )

        if isinstance(embeddings, list):
            raise TypeError(
                "Expected embeddings to be a Tensor or a numpy array, "
                "got a list instead."
            )

        return embeddings.tolist()


class EmbeddingsFactory:
    _instance = None

    @staticmethod
    def create():
        """
        Factory method to create an embeddings instance based on .env config.
        """
        if EmbeddingsFactory._instance is None:
            EmbeddingsFactory._instance = JinaEmbeddings(
                model_name=settings.HUGGINGFACE_EMBEDDINGS_MODEL,
                model_kwargs=settings.HUGGINGFACE_MODEL_KWARGS,
                encode_kwargs=settings.HUGGINGFACE_ENCODE_KWARGS
            )
        return EmbeddingsFactory._instance
