# checklist_manager.py

class ChatChecklist:
    def __init__(self):
        self.state = {
        "got_name": False,
        "picked_topic": False,
        "story_selected": False,
        "story_started": False
        }
        self.data = {
            "child_name": None,
            "topic": None,
            "story_choice": None,
            "current_story_obj": None,  # <-- Add this
            "current_phase": None,      # <-- Add this (e.g., 'entry', 'engagement', 'resolution', 'completed')
            "story_map": {}             # <-- Add this to store the 1 -> "LT01-1" mapping
        }

    def mark_done(self, step):
        if step in self.state:
            self.state[step] = True

    def is_done(self, step):
        return self.state.get(step, False)

    def next_step(self) -> str:
        """Return the next step that isn't done yet."""
        for step, done in self.state.items():
            if not done:
                return step
        return None # all done

    def all_done(self):
        return all(self.state.values())

    def reset(self):
        for step in self.state:
            self.state[step] = False
        # Reset data as well
        self.data = {
            "child_name": None,
            "topic": None,
            "story_choice": None,
            "current_story_obj": None,
            "current_phase": None,
            "story_map": {}
        }