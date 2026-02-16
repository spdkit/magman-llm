#!/usr/bin/env python3
"""
SiliconFlow LLM provider - minimal implementation
"""

import os
from typing import Optional
from .base import BaseLLMProvider


class SiliconFlowProvider(BaseLLMProvider):
    """SiliconFlow minimal implementation"""

    def __init__(self, api_key: str = None, model: str = "deepseek-ai/DeepSeek-V3", debug: bool = False):
        super().__init__(api_key, model, debug)
        self.api_key = api_key or os.getenv('SILICONFLOW_API_KEY')
        if not self.api_key:
            raise ValueError("SiliconFlow API key not set, please set environment variable SILICONFLOW_API_KEY or pass api_key parameter")
    
    def _init_client(self) -> bool:
        """Initialize SiliconFlow client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.siliconflow.cn/v1")
            return True
        except ImportError:
            print("Error: Please install openai library: pip install openai")
            return False
        except Exception as e:
            print(f"Failed to initialize SiliconFlow client: {e}")
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
    
