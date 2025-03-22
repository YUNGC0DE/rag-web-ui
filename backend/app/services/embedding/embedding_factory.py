from app.core.config import settings
from langchain_huggingface import HuggingFaceEmbeddings
# from some_other_module import AnotherEmbeddingClass


class EmbeddingsFactory:
    _instance = None

    @staticmethod
    def create():
        """
        Factory method to create an embeddings instance based on .env config.
        """
        if EmbeddingsFactory._instance is None:
            EmbeddingsFactory._instance = HuggingFaceEmbeddings(
                model_name=settings.HUGGINGFACE_EMBEDDINGS_MODEL,
                model_kwargs=settings.HUGGINGFACE_MODEL_KWARGS,
                encode_kwargs=settings.HUGGINGFACE_ENCODE_KWARGS
            )
        return EmbeddingsFactory._instance
