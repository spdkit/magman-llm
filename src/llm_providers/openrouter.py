#!/usr/bin/env python3
"""
OpenRouter LLM provider
"""

import os
from typing import Optional
from .base import BaseLLMProvider


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter LLM provider"""

    def __init__(self, api_key: str = None, model: str = "anthropic/claude-3.5-sonnet", debug: bool = False):
        """
        Initialize OpenRouter provider

        Args:
            api_key: OpenRouter API key
            model: Model name, defaults to claude-3.5-sonnet
            debug: Whether to enable debug mode
        """
        super().__init__(api_key, model, debug)
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key not set, please set environment variable OPENROUTER_API_KEY or pass api_key parameter")
    
    def _init_client(self) -> bool:
        """Initialize OpenRouter client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/your-repo",  # Optional
                    "X-Title": "VASP Convergence Predictor"  # Optional
                }
            )
            return True
        except ImportError:
            print("Error: Please install openai library: pip install openai")
            return False
        except Exception as e:
            print(f"Failed to initialize OpenRouter client: {e}")
            return False
    
    def _create_completion(self, prompt: str, temperature: float) -> str:
        """Create completion"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional VASP calculation expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            timeout=20
        )
        return response.choices[0].message.content
    
