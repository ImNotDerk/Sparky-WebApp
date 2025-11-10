
class SessionData:
    def __init__(self):
        self.onboarding_data = { # to store data collected during onboarding
            "name": None,
            "chosen_topic": None,
            "chosen_story": None,
            "topic_details": {},
            "story_data": {},
            "story_map": {},  # to map story choices during onboarding
            "chat_session": None
        }
        self.important_conversation_data = { # to store any other important data to retain from the conversation
            "last_hypothesis": None,
            "last_prediction": None,
            "experiment_data": None,
            # to add fixed important data fields as needed
        }

    def reset(self):
        """Resets all data back to the initial state."""
        print("--- SESSION DATA RESET ---")
        # Re-assign the attributes to their default values
        self.onboarding_data = {
            "name": None,
            "chosen_topic": None,
            "chosen_story": None,
            "topic_details": {},
            "story_data": {},
            "story_map": {}  # to map story choices during onboarding
        }
        self.important_conversation_data = {
            "last_hypothesis": None,
            "last_prediction": None,
            "experiment_data": None,
            # to add fixed important data fields as needed
        }