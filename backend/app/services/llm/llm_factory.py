from typing import Optional, List, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import OllamaLLM
from app.core.config import settings

class LLMFactory:
    @staticmethod
    def create(
        provider: Optional[str] = None,
        temperature: float = 0,
        streaming: bool = True,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
    ) -> BaseChatModel:
        print(temperature)
        """
        Create a LLM instance based on the provider
        
        Args:
            provider: The LLM provider to use
            temperature: The temperature to use for generation
            streaming: Whether to stream the response
            callbacks: List of callback handlers to use
        """
        # If no provider specified, use the one from settings
        provider = provider or settings.CHAT_PROVIDER

        if provider.lower() == "openai":
            return ChatOpenAI(
                temperature=temperature,
                streaming=streaming,
                model=settings.OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_API_BASE,
                callbacks=callbacks,
            )
        elif provider.lower() == "deepseek":
            return ChatDeepSeek(
                temperature=temperature,
                streaming=streaming,
                model=settings.DEEPSEEK_MODEL,
                api_key=settings.DEEPSEEK_API_KEY,
                api_base=settings.DEEPSEEK_API_BASE,
                callbacks=callbacks,
            )
        elif provider.lower() == "ollama":
            # Initialize Ollama model
            return OllamaLLM(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_API_BASE,
                temperature=temperature,
                streaming=streaming,
                callbacks=callbacks,
            )
        # Add more providers here as needed
        # elif provider.lower() == "anthropic":
        #     return ChatAnthropic(...)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")