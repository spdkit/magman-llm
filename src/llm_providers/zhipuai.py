#!/usr/bin/env python3
"""
ZhipuAI (GLM) LLM provider - minimal implementation.

Supports both:
- native ZhipuAI SDK
- OpenAI-compatible gateway via environment-provided base_url
"""

import os
from typing import Optional
from .base import BaseLLMProvider


class ZhipuAIProvider(BaseLLMProvider):
    """ZhipuAI minimal implementation"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "glm-4.5-air",
        base_url: str = None,
        debug: bool = False,
    ):
        super().__init__(api_key, model, debug)
        self.api_key = (
            api_key or os.getenv("ZHIPUAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = (
            base_url or os.getenv("ZHIPUAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        )
        if not self.api_key:
            raise ValueError(
                "ZhipuAI API key not set, please set ZHIPUAI_API_KEY/OPENAI_API_KEY or pass api_key parameter"
            )

    def _init_client(self) -> bool:
        """Initialize ZhipuAI client"""
        try:
            if self.base_url:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                if self.debug:
                    print(
                        f"Initialized ZhipuAI provider via OpenAI-compatible gateway: model={self.model}, base_url={self.base_url}"
                    )
                return True

            from zhipuai import ZhipuAI

            self.client = ZhipuAI(api_key=self.api_key)
            if self.debug:
                print(f"Initialized native ZhipuAI provider: model={self.model}")
            return True
        except ImportError as e:
            if self.base_url and "openai" in str(e).lower():
                print("Error: Please install openai library: pip install openai")
                return False
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
                {
                    "role": "system",
                    "content": "You are a professional VASP calculation expert.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content
