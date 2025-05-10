"""
FastAPI Chat Application.

This module implements a FastAPI-based chat application that integrates with OpenAI's API
to provide chat completion functionality. It includes request validation, error handling,
and async API communication.

Usage:
    Run the application using:
    ```
    python app.py
    ```
    The server will start at http://localhost:3000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from openai_service import OpenAIService
from helpers import is_valid_message

app = FastAPI(
    title="Chat API",
    description="A FastAPI application for chat completions using OpenAI",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    """
    Pydantic model for chat request validation.

    Attributes:
        messages (List[Dict[str, str]]): List of chat messages.
            Each message should be a dictionary with 'role' and 'content' keys.
    """
    messages: List[Dict[str, str]]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Process a chat request and return the AI's response.

    This endpoint validates the incoming messages, adds a system prompt,
    and sends the messages to OpenAI for completion.

    Args:
        request (ChatRequest): The chat request containing messages.

    Returns:
        dict: The chat completion response from OpenAI.

    Raises:
        HTTPException: 
            - 400 if messages are invalid or missing
            - 500 if there's an error processing the request

    Example:
        >>> POST /api/chat
        >>> {
        >>>     "messages": [
        >>>         {"role": "user", "content": "Hello"}
        >>>     ]
        >>> }
    """
    # Validate messages
    if not request.messages:
        raise HTTPException(status_code=400, detail="Invalid or missing messages in request body")
    
    if not all(is_valid_message(msg) for msg in request.messages):
        raise HTTPException(status_code=400, detail="Invalid message format in request body")

    openai_service = OpenAIService()
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant who speaks using as fewest words as possible."
    }

    try:
        answer = await openai_service.completion([system_prompt] + request.messages)
        return answer
    except Exception as error:
        print("Error in OpenAI completion:", error)
        raise HTTPException(status_code=500, detail="An error occurred while processing your request")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=3000) 