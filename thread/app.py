"""
FastAPI application implementing a chat system with conversation memory.

This module provides endpoints for chatting with an AI assistant (Alice) that summarizes conversation history
and maintains context between turns. It uses OpenAI's API for generating responses and conversation summaries.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from openai_service import OpenAIService
import uvicorn

app = FastAPI()
openai_service = OpenAIService()
previous_summarization = ""

class ChatRequest(BaseModel):
    """
    Request model for chat endpoint.

    Attributes:
        messages (List[Dict[str, str]]): List of message objects with 'role' and 'content'.
    """
    messages: List[Dict[str, str]]

def create_system_prompt(summarization: str) -> Dict[str, str]:
    """
    Create a system prompt for the assistant, optionally including a conversation summary.

    Args:
        summarization (str): The summary of the conversation so far.

    Returns:
        Dict[str, str]: The system prompt for the assistant.
    """
    return {
        "role": "system",
        "content": f"""You are Alice, a helpful assistant who speaks using as few words as possible. 
        {"Here is a summary of the conversation so far: \n    <conversation_summary>\n      "+summarization+"\n    </conversation_summary>" if summarization else ""} 
        Let's chat!"""
    }

def generate_summarization(user_message: Dict[str, str], assistant_response: Dict[str, str]) -> str:
    """
    Generate a summary of the conversation using the latest user message and assistant response.

    Args:
        user_message (Dict[str, str]): The latest user message.
        assistant_response (Dict[str, str]): The latest assistant response.

    Returns:
        str: The updated conversation summary.
    """
    summarization_prompt = {
        "role": "system",
        "content": f"""Please summarize the following conversation in a concise manner:
<previous_summary>{previous_summarization or "No previous summary"}</previous_summary>
<current_turn> User: {user_message['content']}\nAssistant: {assistant_response['content']} </current_turn>
"""
    }

    response = openai_service.completion(
        [
            summarization_prompt,
            {"role": "user", "content": "Please create/update our conversation summary."}
        ],
        "gpt-4o-mini",
        False
    )
    return response["choices"][0]["message"]["content"]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Endpoint for chatting with the assistant.

    Args:
        request (ChatRequest): The chat request containing user messages.

    Returns:
        dict: The assistant's response.

    Raises:
        HTTPException: If no messages are provided or an error occurs.
    """
    global previous_summarization
    
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    try:
        system_prompt = create_system_prompt(previous_summarization)
        assistant_response = openai_service.completion(
            [system_prompt, request.messages[0]], "gpt-4o", False
        )

        previous_summarization = generate_summarization(
            request.messages[0], 
            assistant_response["choices"][0]["message"]
        )

        return assistant_response
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

@app.post("/api/demo")
async def demo():
    """
    Demo endpoint to simulate a multi-turn conversation with the assistant.

    Returns:
        dict: The assistant's response to the last demo message.

    Raises:
        HTTPException: If an error occurs during the conversation.
    """
    global previous_summarization
    demo_messages = [
        {"content": "Hi! I'm Adam", "role": "user"},
        {"content": "How are you?", "role": "user"},
        {"content": "Do you know my name?", "role": "user"}
    ]

    assistant_response = None

    for message in demo_messages:
        print("--- NEXT TURN ---")
        print("Adam:", message["content"])

        try:
            system_prompt = create_system_prompt(previous_summarization)
            assistant_response = openai_service.completion(
                [system_prompt, message], "gpt-4o", False
            )
            print("Alice:", assistant_response["choices"][0]["message"]["content"])

            previous_summarization = generate_summarization(
                message, 
                assistant_response["choices"][0]["message"]
            )
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error))

    return assistant_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)