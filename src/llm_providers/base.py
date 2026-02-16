#!/usr/bin/env python3
"""
Base class for LLM providers - minimal necessary refactoring
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
import time
import multiprocessing
from multiprocessing import Process, Queue
import os


class BaseLLMProvider(ABC):
    """Minimalist LLM provider base class - solves code duplication"""

    def __init__(self, api_key: str, model: str, debug: bool = False):
        self.api_key = api_key
        self.model = model
        self.debug = debug
        self.client = None

    def call_api(self, prompt: str, temperature: float = 0.1,
                 timeout: int = 20, max_retries: int = 2) -> Optional[str]:
        """Generic API call - shared by all providers"""
        if not self.client:
            if not self._init_client():
                return None

        if self.debug:
            print(f"\n=== DEBUG: API call started ===")
            print(f"Provider: {self.__class__.__name__.replace('Provider', '')}")
            print(f"Model: {self.model}")

        def _make_api_call():
            return self._create_completion(prompt, temperature)

        return self._call_with_timeout_and_retry(
            _make_api_call, timeout=timeout, max_retries=max_retries
        )

    @abstractmethod
    def _init_client(self) -> bool:
        """Initialize client - different for each provider"""
        pass

    @abstractmethod
    def _create_completion(self, prompt: str, temperature: float) -> str:
        """Create completion - different for each provider"""
        pass

    def _call_with_timeout_and_retry(self, api_call_func: Callable, *args, **kwargs) -> Optional[str]:
        """
        Generic API call logic with timeout and retry

        Args:
            api_call_func: API call function
            *args, **kwargs: Arguments passed to the API call function

        Returns:
            API response or None
        """
        timeout = kwargs.pop('timeout', 20)
        max_retries = kwargs.pop('max_retries', 2)
        retry_delay = 30

        for attempt in range(max_retries):
            if self.debug:
                print(f"Attempt {attempt+1} of API call...")

            start_time = time.time()

            try:
                # Use process-level timeout control
                result = self._call_with_process_timeout(api_call_func, *args, timeout=timeout, **kwargs)

                elapsed_time = time.time() - start_time

                if self.debug:
                    print(f"=== DEBUG: API call successful ===")
                    print(f"Actual time: {elapsed_time:.2f}s")
                    print(f"Response length: {len(result) if result else 0} characters")
                    if result:
                        print(f"Response content:\n{result[:500]}...")
                    print("=" * 50)

                return result

            except Exception as e:
                elapsed_time = time.time() - start_time

                # Check if it's a timeout error
                is_timeout = any(keyword in str(e).lower() for keyword in ['timeout', 'timed out'])

                if is_timeout:
                    if self.debug:
                        print(f"Attempt {attempt+1} timed out: {e} (elapsed: {elapsed_time:.2f}s)")

                    if attempt == max_retries - 1:
                        print(f"API call timed out (retried {max_retries} times, total time: {elapsed_time:.2f}s)")
                        return None
                    print(f"API call timed out, retry {attempt+1}...")
                else:
                    if self.debug:
                        print(f"Attempt {attempt+1} failed: {str(e)} (elapsed: {elapsed_time:.2f}s)")

                    if attempt == max_retries - 1:
                        print(f"API call failed (retried {max_retries} times): {str(e)}")
                        return None
                    print(f"API call failed, retry {attempt+1}: {str(e)}")

            # Retry delay
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        return None

    def _call_with_process_timeout(self, api_call_func, *args, **kwargs) -> Optional[str]:
        """
        Call API function with process-level timeout control

        Args:
            api_call_func: API call function
            *args, **kwargs: Arguments passed to the API call function

        Returns:
            API response or None
        """
        timeout = kwargs.pop('timeout', 20)

        # Create inter-process communication queue
        result_queue = Queue()

        def _worker():
            """Worker process function"""
            try:
                result = api_call_func(*args, **kwargs)
                result_queue.put(('success', result))
            except Exception as e:
                result_queue.put(('error', str(e)))

        # Start worker process
        process = Process(target=_worker)
        process.start()

        # Wait for process to complete or timeout
        process.join(timeout=timeout)

        if process.is_alive():
            # Process timed out, terminate it
            process.terminate()
            process.join()
            raise TimeoutError(f"API call timed out ({timeout}s)")

        # Get result
        if not result_queue.empty():
            result = result_queue.get()
            if result is not None:
                try:
                    status, data = result
                    if status == 'success':
                        return data
                    else:
                        raise Exception(data)
                except (TypeError, ValueError) as e:
                    raise Exception(f"Result format error: {result}, error: {e}")

        raise Exception("API call failed, no result returned")
