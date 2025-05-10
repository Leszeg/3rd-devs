"""
FastAPI Chat Application with Streaming Support.

This module implements a FastAPI-based chat application that integrates with OpenAI's API
to provide chat completion functionality with streaming support. It includes request
validation, error handling, and both streaming and non-streaming responses.

Usage:
    Run the application using:
    ```
    python app.py
    ```
    The server will start at http://localhost:3000
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from openai_service import OpenAIService
from helpers import is_valid_message
import uuid
import json
import uvicorn
from datetime import datetime

app = FastAPI(
    title="Streaming Chat API",
    description="A FastAPI application for chat completions using OpenAI with streaming support",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    """
    Pydantic model for chat request validation.

    Attributes:
        messages (List[Dict[str, str]]): List of chat messages.
            Each message should be a dictionary with 'role' and 'content' keys.
        stream (bool, optional): Whether to stream the response. Defaults to False.
    """
    messages: List[Dict[str, str]]
    stream: Optional[bool] = False

async def stream_response(messages: List[Dict[str, str]], conversation_uuid: str):
    """
    Generate streaming response for chat completion.

    Args:
        messages (List[Dict[str, str]]): List of messages for the chat completion.
        conversation_uuid (str): Unique identifier for the conversation.

    Yields:
        str: SSE formatted JSON string containing the chat completion chunk.
    """
    # Send initial chunk
    starting_chunk = {
        "id": f"chatcmpl-{int(datetime.now().timestamp())}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": "gpt-4",
        "system_fingerprint": f"fp_{uuid.uuid4().hex[:10]}",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": "starting response"},
            "logprobs": None,
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(starting_chunk)}\n\n"
    
    try:
        async with OpenAIService() as openai_service:
            result = await openai_service.completion(messages, stream=True)
            async for chunk in result:
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"
    except Exception as error:
        print("Error in streaming response:", error)
        error_chunk = {
            "id": f"chatcmpl-{int(datetime.now().timestamp())}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": "gpt-4",
            "system_fingerprint": f"fp_{uuid.uuid4().hex[:10]}",
            "choices": [{
                "index": 0,
                "delta": {"content": "An error occurred during streaming"},
                "logprobs": None,
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Process a chat request and return the AI's response.

    This endpoint validates the incoming messages, adds a system prompt,
    and sends the messages to OpenAI for completion. Supports both
    streaming and non-streaming responses.

    Args:
        request (ChatRequest): The chat request containing messages and stream flag.

    Returns:
        Union[StreamingResponse, JSONResponse]: 
            - StreamingResponse for streaming requests
            - JSONResponse for non-streaming requests

    Raises:
        HTTPException: 
            - 400 if messages are invalid or missing
            - 500 if there's an error processing the request

    Example:
        >>> POST /api/chat
        >>> {
        >>>     "messages": [
        >>>         {"role": "user", "content": "Hello"}
        >>>     ],
        >>>     "stream": true
        >>> }
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Invalid or missing messages in request body")
    
    if not all(is_valid_message(msg) for msg in request.messages):
        raise HTTPException(status_code=400, detail="Invalid message format in request body")

    conversation_uuid = str(uuid.uuid4())
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant who speaks using as fewest words as possible."
    }
    
    messages = [system_prompt] + request.messages

    try:
        if request.stream:
            return StreamingResponse(
                stream_response(messages, conversation_uuid),
                media_type="text/event-stream",
                headers={
                    "X-Conversation-UUID": conversation_uuid,
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
        else:
            async with OpenAIService() as openai_service:
                response = await openai_service.completion(messages)
                return JSONResponse({
                    **response.model_dump(),
                    "conversationUUID": conversation_uuid
                })
    except Exception as error:
        print("Error in OpenAI completion:", error)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=3000) 