import time
import uuid
from typing import Dict, List


class ConversationManager:
    """
    A class to manage in-memory chat conversations with only question-answer pairs.

    Attributes:
    -----------
    conversations : Dict[str, Dict]
        A dictionary to store conversation data. Each conversation is identified by a unique conversation_id.

    max_history : int
        The maximum number of question-answer pairs to store in the conversation history.

    timeout : int
        The time in seconds after which a conversation expires due to inactivity.
    """

    def __init__(self, max_history: int = 10, timeout: int = 900):
        """
        Initializes the ConversationManager with a given conversation timeout and history size.

        Parameters:
        -----------
        max_history : int
            The maximum number of question-answer pairs to store in the conversation history (default is 10).

        timeout : int
            The conversation expiration time in seconds (default is 900 seconds or 15 minutes).
        """
        self.conversations: Dict[str, Dict] = {}
        self.max_history = max_history
        self.timeout = timeout

    def create_conversation(self, conversation_id: str = None) -> str:
        """
        Creates a new conversation with a given conversation_id or generates a new one.

        Parameters:
        -----------
        conversation_id : str, optional
            The conversation identifier. If not provided, a new UUID will be generated.

        Returns:
        --------
        str:
            The conversation_id for the new conversation.
        """
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        self.conversations[conversation_id] = {
            "qa_pairs": [],
            "last_active": time.time(),
        }
        return conversation_id

    def get_qa_pairs(self, conversation_id: str) -> List[tuple]:
        """
        Retrieves the list of question-answer pairs for a given conversation_id.
        If the conversation doesn't exist, it creates a new one.

        Parameters:
        -----------
        conversation_id : str
            The unique conversation identifier.

        Returns:
        --------
        List[tuple]:
            A list of tuples where each tuple contains a question and its corresponding answer.
        """
        if conversation_id not in self.conversations:
            self.create_conversation(conversation_id)
        return self.conversations[conversation_id]["qa_pairs"]

    def add_message_pair(
        self, conversation_id: str, question: str, answer: str
    ) -> None:
        """
        Adds a question-answer pair to an existing conversation's history.
        If the conversation doesn't exist, it creates a new one.

        Parameters:
        -----------
        conversation_id : str
            The unique conversation identifier.

        question : str
            The user's query (question).

        answer : str
            The assistant's response (answer).
        """
        if conversation_id not in self.conversations:
            self.create_conversation(conversation_id)

        self.conversations[conversation_id]["qa_pairs"].append((question, answer))
        if len(self.conversations[conversation_id]["qa_pairs"]) > self.max_history:
            self.conversations[conversation_id]["qa_pairs"].pop(0)
        self.conversations[conversation_id]["last_active"] = time.time()

    def clear_inactive_conversations(self) -> None:
        """
        A background task that periodically checks and clears inactive conversations
        that have exceeded the timeout period.
        """
        current_time = time.time()
        inactive_conversations = [
            cid
            for cid, data in self.conversations.items()
            if current_time - data["last_active"] > self.timeout
        ]
        for cid in inactive_conversations:
            del self.conversations[cid]
