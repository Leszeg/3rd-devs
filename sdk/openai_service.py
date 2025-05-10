"""
OpenAI service integration module.

This module provides a service class for interacting with the OpenAI API,
specifically for handling chat completions.
"""

from typing import List, Dict, Any
from openai import OpenAI

class OpenAIService:
    """
    A service class for handling OpenAI API interactions.

    This class provides methods for creating chat completions using the OpenAI API.
    It initializes with a default OpenAI client and handles API communication.

    Attributes:
        openai (OpenAI): The OpenAI client instance used for API calls.
    """

    def __init__(self):
        """
        Initialize the OpenAIService with a new OpenAI client.

        Note:
            Requires OPENAI_API_KEY environment variable to be set.
        """
        self.openai = OpenAI()
    
    async def completion(self, messages: List[Dict[str, str]], model: str = "gpt-4") -> Any:
        """
        Create a chat completion using the OpenAI API.

        Args:
            messages (List[Dict[str, str]]): List of message objects.
                Each message should have 'role' and 'content' keys.
            model (str, optional): The OpenAI model to use. Defaults to "gpt-4".

        Returns:
            Any: The chat completion response from OpenAI.

        Raises:
            Exception: If there's an error during the API call.

        Example:
            >>> messages = [{"role": "user", "content": "Hello"}]
            >>> response = await completion(messages)
        """
        try:
            chat_completion = await self.openai.chat.completions.create(
                messages=messages,
                model=model
            )
            return chat_completion
        except Exception as error:
            print("Error in OpenAI completion:", error)
            raise 