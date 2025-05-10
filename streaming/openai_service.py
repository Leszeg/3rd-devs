"""
OpenAI service integration module with streaming support.

This module provides an asynchronous service class for interacting with the OpenAI API,
specifically for handling chat completions with streaming capability using AsyncOpenAI client.
"""

from typing import List, Dict, Union, AsyncGenerator
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

class OpenAIService:
    """
    An asynchronous service class for handling OpenAI API interactions with streaming support.

    This class provides asynchronous methods for creating chat completions using the OpenAI API,
    supporting both regular and streaming responses using AsyncOpenAI client.

    Attributes:
        openai (AsyncOpenAI): The AsyncOpenAI client instance used for API calls.
            This client provides native async/await support for all API operations.
    """

    def __init__(self):
        """
        Initialize the OpenAIService with a new AsyncOpenAI client.

        Note:
            Requires OPENAI_API_KEY environment variable to be set.
            Uses AsyncOpenAI client for better async/await support and performance.
        """
        self.openai = AsyncOpenAI()
    
    async def completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        stream: bool = False
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """
        Create an asynchronous chat completion using the OpenAI API with optional streaming.

        This method uses AsyncOpenAI client to make non-blocking API calls. For streaming responses,
        it returns an async generator that can be iterated over using async for.

        Args:
            messages (List[Dict[str, str]]): List of message objects.
                Each message should have 'role' and 'content' keys.
            model (str, optional): The OpenAI model to use. Defaults to "gpt-4".
            stream (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
                If stream=False, returns a ChatCompletion object.
                If stream=True, returns an AsyncGenerator yielding ChatCompletionChunks.

        Raises:
            Exception: If there's an error during the API call.

        Example:
            >>> async with OpenAIService() as service:
            >>>     # For non-streaming response
            >>>     messages = [{"role": "user", "content": "Hello"}]
            >>>     response = await service.completion(messages)
            >>>     
            >>>     # For streaming response
            >>>     async for chunk in await service.completion(messages, stream=True):
            >>>         print(chunk.choices[0].delta.content)
        """
        try:
            chat_completion = await self.openai.chat.completions.create(
                messages=messages,
                model=model,
                stream=stream
            )
            return chat_completion
        
        except Exception as error:
            print("Error in OpenAI completion:", error)
            raise

    async def __aenter__(self):
        """
        Async context manager entry point.

        Returns:
            OpenAIService: The service instance.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit point.

        Args:
            exc_type: The type of the exception that was raised, if any.
            exc_val: The instance of the exception that was raised, if any.
            exc_tb: The traceback of the exception that was raised, if any.
        """
        await self.openai.close() 