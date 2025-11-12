from checklist_manager import ChatChecklist
from session_data_manager import SessionData
from google.genai import types

class ChatSessionManager:
    def __init__(self):
        self.sessions = {} # { "session_id_123": {"checklist": ..., "history": ...} }

    def get_or_create_session(self, session_id: str) -> dict:
        """Gets an existing session or creates a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "checklist": ChatChecklist(),
                "session_data": SessionData(),
                "history": []
            }
        return self.sessions[session_id]

    def get_checklist(self, session_id: str) -> ChatChecklist:
        return self.get_or_create_session(session_id)["checklist"]

    def get_session_data(self, session_id: str) -> SessionData:
        return self.get_or_create_session(session_id)["session_data"]

    def get_history(self, session_id: str) -> list[types.Content]:
        """Returns the conversation history for the given session."""
        session = self.get_or_create_session(session_id)
        return session["history"]

    def update_history(self, session_id: str, message: types.Content):
        """Updates the conversation history for the given session."""
        session = self.get_or_create_session(session_id)
        session["history"].append(message)

    def save_session_data(self, session_id: str, checklist: ChatChecklist, session_data: SessionData, history: list[types.Content]):
        """Saves the checklist and history for the given session."""
        self.sessions[session_id] = {
            "checklist": checklist,
            "session_data": session_data,
            "history": history
        }

    def reset_session(self, session_id: str):
        """Resets the session data for the given session."""
        if session_id in self.sessions:
            del self.sessions[session_id]