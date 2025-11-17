
class SessionData:
    def __init__(self):
        self.onboarding_data = { # to store data collected during onboarding
            "name": None,
            "chosen_topic": None,
            "topic_details": {},
            "topic_stories": {}, # store actual story data
            "topic_stories_list": {}, # store list of stories
            "story_data": {}, # chosen story data
            "chat_session": None,
            "end_goal": None # lesson to be learned from the story
        }
        self.important_conversation_data = { # to store any other important data to retain from the conversation
            "initial_story_narration": None,
            "hypothesis_question": None,
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
            "topic_stories": {}, # store actual story data
            "topic_stories_list": {}, # store list of stories
            "chat_session": None,
            "end_goal": None # lesson to be learned from the story
        }
        self.important_conversation_data = {
            "initial_story_narration": None,
            "hypothesis_question": None,
            "last_hypothesis": None,
            "last_prediction": None,
            "experiment_data": None,
            # to add fixed important data fields as needed
        }