# app/utils/session_manager.py

import time
import uuid
import asyncio
from typing import Dict, List, Optional


class SessionManager:
    """
    A class to manage in-memory chat sessions with a timeout mechanism.

    Attributes:
    -----------
    session_store : Dict[str, Dict]
        A dictionary to store session data. Each session is identified by a unique session_id.

    timeout : int
        The time in seconds after which a session expires due to inactivity.
    """

    def __init__(self, timeout: int = 900):
        """
        Initializes the SessionManager with a given session timeout.

        Parameters:
        -----------
        timeout : int
            The session expiration time in seconds (default is 900 seconds or 15 minutes).
        """
        self.session_store: Dict[str, Dict] = {}
        self.timeout = timeout

    def create_session(self) -> str:
        """
        Creates a new session with a unique session_id (UUID) and initializes its data.

        Returns:
        --------
        str:
            The generated unique session_id for the new session.
        """
        session_id = str(uuid.uuid4())
        self.session_store[session_id] = {"messages": [], "last_active": time.time()}
        return session_id

    def store_message(self, session_id: str, role: str, message: str) -> None:
        """
        Stores a message in an existing session's chat history.

        Parameters:
        -----------
        session_id : str
            The unique session identifier.
        role : str
            The role of the sender (either "user" or "bot").
        message : str
            The message content to be stored.

        Raises:
        -------
        KeyError:
            If the session_id does not exist in the session store.
        """
        if session_id in self.session_store:
            self.session_store[session_id]["messages"].append(
                {"role": role, "message": message}
            )
            self.session_store[session_id]["last_active"] = time.time()
        else:
            raise KeyError(f"Session ID {session_id} not found.")

    def get_history(self, session_id: str) -> Optional[List[Dict[str, str]]]:
        """
        Retrieves the chat history for a given session_id.

        Parameters:
        -----------
        session_id : str
            The unique session identifier.

        Returns:
        --------
        Optional[List[Dict[str, str]]]:
            A list of dictionaries containing role and message pairs for the session.
            If the session doesn't exist, returns None.
        """
        if session_id in self.session_store:
            return self.session_store[session_id]["messages"]
        return None

    def end_session(self, session_id: str) -> None:
        """
        Ends a session and removes it from the session store.

        Parameters:
        -----------
        session_id : str
            The unique session identifier.

        Raises:
        -------
        KeyError:
            If the session_id does not exist in the session store.
        """
        if session_id in self.session_store:
            del self.session_store[session_id]
        else:
            raise KeyError(f"Session ID {session_id} not found.")

    async def clear_inactive_sessions(self) -> None:
        """
        A background task that periodically checks and clears inactive sessions
        that have exceeded the timeout period.

        This method should be run as a background task in the event loop.
        """
        while True:
            current_time = time.time()
            sessions_to_clear = [
                sid
                for sid, data in self.session_store.items()
                if current_time - data["last_active"] > self.timeout
            ]
            for sid in sessions_to_clear:
                del self.session_store[sid]
            await asyncio.sleep(300)  # Check every 5 minutes
