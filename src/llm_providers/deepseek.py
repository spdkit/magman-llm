#!/usr/bin/env python3
"""
DeepSeek API provider implementation
"""

import os
import time
from typing import Optional
from openai import OpenAI
from .base import BaseLLMProvider


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API provider"""

    def __init__(self, model: str = "deepseek-chat", api_key: str = None,
                 base_url: str = "https://api.deepseek.com", debug: bool = False, **kwargs):
        """
        Initialize DeepSeek provider

        Args:
            model: Model name
            api_key: API key
            base_url: API base URL
            debug: Debug mode
        """
        # Call parent class initialization
        super().__init__(api_key, model, debug)

        # Set DeepSeek-specific parameters
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("DeepSeek API key not provided, please set --api-key parameter or DEEPSEEK_API_KEY environment variable")
    
    def _init_client(self) -> bool:
        """Initialize DeepSeek client"""
        try:
            # Initialize OpenAI client (compatible with DeepSeek API)
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            if self.debug:
                print(f"Initialized DeepSeek provider: model={self.model}, base_url={self.base_url}")

            return True

        except Exception as e:
            print(f"Failed to initialize DeepSeek client: {e}")
            return False
    
    def _create_completion(self, prompt: str, temperature: float) -> str:
        """Create DeepSeek completion"""
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False
        )
        
        return response.choices[0].message.content