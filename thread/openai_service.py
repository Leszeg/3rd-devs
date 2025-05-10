"""
OpenAI service module for handling API interactions.

This module provides a service class for interacting with the OpenAI API, specifically for chat completions.
"""

from openai import OpenAI
from typing import Dict, List, Any

class OpenAIService:
    """
    Service class for interacting with the OpenAI API.
    """
    def __init__(self):
        """
        Initialize OpenAI service with default client.
        """
        self.openai = OpenAI()

    def completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a chat completion using the OpenAI API.

        Args:
            messages (List[Dict[str, str]]): List of message objects for the conversation.
            model (str, optional): The model to use for completion. Defaults to "gpt-4".
            stream (bool, optional): Whether to use streaming responses. Defaults to False.

        Returns:
            Dict[str, Any]: The API response containing the assistant's message and metadata.

        Raises:
            Exception: If the OpenAI API call fails.
        """
        try:
            chat_completion = self.openai.chat.completions.create(
                messages=messages,
                model=model,
                stream=stream
            )

            return {
                "choices": [{
                    "message": {
                        "role": chat_completion.choices[0].message.role,
                        "content": chat_completion.choices[0].message.content
                    },
                    "index": chat_completion.choices[0].index,
                }],
                "model": chat_completion.model,
                "usage": chat_completion.usage.model_dump()
            }

        except Exception as error:
            print(f"Error in OpenAI completion: {str(error)}")
            raise error