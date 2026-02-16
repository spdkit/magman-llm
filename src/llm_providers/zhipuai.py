#!/usr/bin/env python3
"""
ZhipuAI (GLM) LLM provider - minimal implementation
"""

import os
from typing import Optional
from .base import BaseLLMProvider


class ZhipuAIProvider(BaseLLMProvider):
    """ZhipuAI minimal implementation"""

    def __init__(self, api_key: str = None, model: str = "glm-4.5-air", debug: bool = False):
        super().__init__(api_key, model, debug)
        self.api_key = api_key or os.getenv('ZHIPUAI_API_KEY')
        if not self.api_key:
            raise ValueError("ZhipuAI API key not set, please set environment variable ZHIPUAI_API_KEY or pass api_key parameter")
    
    def _init_client(self) -> bool:
        """Initialize ZhipuAI client"""
        try:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=self.api_key)
            return True
        except ImportError:
            print("Error: Please install zhipuai library: pip install zhipuai")
            return False
        except Exception as e:
            print(f"Failed to initialize ZhipuAI client: {e}")
            return False
    
    def _create_completion(self, prompt: str, temperature: float) -> str:
        """Create completion"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional VASP calculation expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    
