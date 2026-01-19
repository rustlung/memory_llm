"""Short-term memory management for chat history."""
from typing import List, Dict
from collections import defaultdict, deque


class ShortMemory:
    """Manages short-term memory (last N messages) per chat."""
    
    def __init__(self, max_size: int = 10):
        """
        Initialize short memory.
        
        Args:
            max_size: Maximum number of messages to store per chat
        """
        self.max_size = max_size
        # chat_id -> deque of messages
        self._storage: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_size))
    
    def add_message(self, chat_id: int, role: str, content: str) -> None:
        """
        Add a message to chat history.
        
        Args:
            chat_id: Telegram chat ID
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        self._storage[chat_id].append({
            "role": role,
            "content": content
        })
    
    def get_history(self, chat_id: int) -> List[Dict[str, str]]:
        """
        Get chat history for a specific chat.
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return list(self._storage[chat_id])
    
    def clear(self, chat_id: int) -> None:
        """
        Clear chat history for a specific chat.
        
        Args:
            chat_id: Telegram chat ID
        """
        if chat_id in self._storage:
            self._storage[chat_id].clear()
    
    def get_size(self, chat_id: int) -> int:
        """
        Get number of messages in chat history.
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            Number of messages
        """
        return len(self._storage[chat_id])
