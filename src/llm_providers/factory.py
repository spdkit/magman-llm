#!/usr/bin/env python3
"""
LLM Provider Factory
"""

from typing import Dict, Type
from .base import BaseLLMProvider
from .zhipuai import ZhipuAIProvider
from .siliconflow import SiliconFlowProvider
from .openrouter import OpenRouterProvider
from .litellm_provider import LiteLLMProvider
from .deepseek import DeepSeekProvider


class LLMProviderFactory:
    """LLM Provider Factory"""

    _providers: Dict[str, Type[BaseLLMProvider]] = {
        'zhipuai': ZhipuAIProvider,
        'siliconflow': SiliconFlowProvider,
        'openrouter': OpenRouterProvider,
        'litellm': LiteLLMProvider,
        'deepseek': DeepSeekProvider,
    }

    @classmethod
    def create_provider(cls, provider_name: str, **kwargs) -> BaseLLMProvider:
        """
        Create LLM provider instance

        Args:
            provider_name: Provider name
            **kwargs: Arguments passed to provider constructor

        Returns:
            LLM provider instance

        Raises:
            ValueError: Unsupported provider
        """
        if provider_name not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(f"Unsupported provider: {provider_name}. Available providers: {available}")

        return cls._providers[provider_name](**kwargs)

    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available providers"""
        return list(cls._providers.keys())



# Convenience function
def create_llm_provider(provider_name: str, **kwargs) -> BaseLLMProvider:
    """Convenience function to create LLM provider"""
    return LLMProviderFactory.create_provider(provider_name, **kwargs)
