"""
Helper functions for message validation in the streaming chat API.

This module provides utility functions for validating the structure and content
of chat messages used in the streaming API.
"""

def is_valid_message(message: dict) -> bool:
    """
    Validate if a message has the required structure and content types.

    Args:
        message (dict): The message object to validate.
            Expected format: {
                'role': str,
                'content': str
            }

    Returns:
        bool: True if the message is valid, False otherwise.

    Example:
        >>> msg = {'role': 'user', 'content': 'Hello'}
        >>> is_valid_message(msg)
        True
    """
    return (isinstance(message, dict) and
            'role' in message and
            'content' in message and
            isinstance(message['content'], str)) 