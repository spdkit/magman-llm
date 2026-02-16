#!/usr/bin/env python3
"""
LiteLLM unified provider implementation
Supports 100+ LLM providers, including Kimi, DeepSeek, etc.
"""

import os
from typing import Optional, List, Dict, Any
from .base import BaseLLMProvider


class LiteLLMProvider(BaseLLMProvider):
    """LiteLLM unified provider, supports multiple LLM platforms"""

    # Provider to environment variable mapping
    PROVIDER_CONFIGS = {
        'moonshot': {
            'env_key': 'MOONSHOT_API_KEY',
            'model_prefix': 'moonshot/',
            'base_url': 'https://api.moonshot.cn/v1'
        },
        'deepseek': {
            'env_key': 'DEEPSEEK_API_KEY', 
            'model_prefix': 'deepseek/',
            'base_url': 'https://api.deepseek.com'
        },
        'volcano': {
            'env_key': 'VOLCANO_ACCESS_KEY',
            'model_prefix': 'volcano/',
            'base_url': 'https://ark.cn-beijing.volces.com/api/v3'
        },
        'dashscope': {
            'env_key': 'DASHSCOPE_API_KEY',
            'model_prefix': 'dashscope/',
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        }
    }
    
    def __init__(self, api_key: str = None, model: str = None, debug: bool = False):
        """
        Initialize LiteLLM provider

        Args:
            api_key: API key (optional, environment variable takes priority)
            model: Model name, format is "provider/model-name"
            debug: Whether to enable debug mode
        """
        if not model or '/' not in model:
            raise ValueError("LiteLLM provider requires model format 'provider/model-name'")

        provider_name = model.split('/')[0]
        if provider_name not in self.PROVIDER_CONFIGS:
            raise ValueError(f"Unsupported LiteLLM provider: {provider_name}")

        config = self.PROVIDER_CONFIGS[provider_name]

        # Set API key
        self.api_key = api_key or os.getenv(config['env_key'])
        if not self.api_key:
            raise ValueError(f"{provider_name} API key not set, please set environment variable {config['env_key']} or pass api_key parameter")
        
        super().__init__(self.api_key, model, debug)
        self.provider_name = provider_name
        self.config = config
    
    def _init_client(self) -> bool:
        """Initialize LiteLLM client"""
        try:
            import litellm
            # LiteLLM does not require explicit client initialization, uses environment variables

            # Set provider-specific environment variables
            os.environ[self.config['env_key']] = self.api_key

            # Optional: set custom endpoint
            if 'base_url' in self.config:
                endpoint_env = f"{self.provider_name.upper()}_API_BASE"
                os.environ[endpoint_env] = self.config['base_url']

            return True
        except ImportError:
            print("Error: Please install litellm library: pip install litellm")
            return False
        except Exception as e:
            print(f"Failed to initialize LiteLLM client: {e}")
            return False
    
    def _create_completion(self, prompt: str, temperature: float) -> str:
        """Create completion"""
        import litellm

        extra_params = {'drop_params': True}
        if self.provider_name == 'dashscope':
            extra_params.update({'enable_thinking': False, 'thinking': {'enabled': False}})

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional VASP calculation expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            timeout=20,
            **extra_params
        )

        message = response.choices[0].message
        return message.content or getattr(message, 'reasoning_content', '')
    
    
    @staticmethod
    def get_supported_providers() -> Dict[str, str]:
        """Get list of supported providers and environment variable names"""
        return {
            'moonshot': 'MOONSHOT_API_KEY',
            'deepseek': 'DEEPSEEK_API_KEY',
            'volcano': 'VOLCANO_ACCESS_KEY', 
            'dashscope': 'DASHSCOPE_API_KEY'
        }