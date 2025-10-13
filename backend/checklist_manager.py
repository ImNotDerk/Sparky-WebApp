class ChatChecklist:
    def __init__(self):
        self.state = {
        "got_name": False,
        "picked_topic": False,
        "story_started": False
        }
        self.data = {
            "child_name": None,
            "topic": None
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