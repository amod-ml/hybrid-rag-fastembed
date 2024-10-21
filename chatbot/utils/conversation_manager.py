import time
import uuid
from typing import Dict, List, Optional
from collections import deque


class ConversationManager:
    """
    A class to manage in-memory chat conversations with a timeout mechanism.

    Attributes:
    -----------
    conversations : Dict[str, Dict]
        A dictionary to store conversation data. Each conversation is identified by a unique conversation_id.

    max_history : int
        The maximum number of messages to store in the conversation history.

    timeout : int
        The time in seconds after which a conversation expires due to inactivity.
    """

    def __init__(self, max_history: int = 10, timeout: int = 900):
        """
        Initializes the ConversationManager with a given conversation timeout and history size.

        Parameters:
        -----------
        max_history : int
            The maximum number of messages to store in the conversation history (default is 10).

        timeout : int
            The conversation expiration time in seconds (default is 900 seconds or 15 minutes).
        """
        self.conversations: Dict[str, Dict] = {}
        self.max_history = max_history
        self.timeout = timeout

    def create_conversation(self) -> str:
        """
        Creates a new conversation with a unique conversation_id (UUID) and initializes its data.

        Returns:
        --------
        str:
            The generated unique conversation_id for the new conversation.
        """
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = {
            "history": deque(maxlen=self.max_history),
            "last_active": time.time()
        }
        return conversation_id

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """
        Adds a message to an existing conversation's chat history.

        Parameters:
        -----------
        conversation_id : str
            The unique conversation identifier.

        role : str
            The role of the sender (either "user" or "bot").

        content : str
            The message content to be stored.

        Raises:
        -------
        KeyError:
            If the conversation_id does not exist in the conversation store.
        """
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["history"].append({"role": role, "content": content})
            self.conversations[conversation_id]["last_active"] = time.time()
        else:
            raise KeyError(f"Conversation ID {conversation_id} not found.")

    def get_history(self, conversation_id: str) -> Optional[List[Dict[str, str]]]:
        """
        Retrieves the chat history for a given conversation_id.

        Parameters:
        -----------
        conversation_id : str
            The unique conversation identifier.

        Returns:
        --------
        Optional[List[Dict[str, str]]]:
            A list of dictionaries containing role and content pairs for the conversation.
            If the conversation doesn't exist, returns None.
        """
        if conversation_id in self.conversations:
            return list(self.conversations[conversation_id]["history"])
        return None

    def clear_inactive_conversations(self) -> None:
        """
        A background task that periodically checks and clears inactive conversations
        that have exceeded the timeout period.

        This method should be run as a background task in the event loop.
        """
        current_time = time.time()
        inactive_conversations = [
            cid for cid, data in self.conversations.items()
            if current_time - data["last_active"] > self.timeout
        ]
        for cid in inactive_conversations:
            del self.conversations[cid]
