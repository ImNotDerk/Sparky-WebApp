class ChecklistState:
    """A generic class to manage a checklist of steps."""

    def __init__(self, steps: list[str]):
        # We use the list of steps to build the initial state dictionary
        self.steps = steps
        self.state = {step: False for step in self.steps}

    def get_current_phase(self) -> str | None:
        """Finds the first step in the state that is not yet 'True'."""
        for step, done in self.state.items():
            if not done:
                return step
        return None  # All steps are done

    def mark_done(self, step: str):
        """Marks a specific step as True."""
        if step in self.state:
            self.state[step] = True
        else:
            print(f"Warning: Step '{step}' not found in this checklist.")

    def mark_undone(self, step: str):
        """Marks a specific step as False."""
        if step in self.state:
            self.state[step] = False
        else:
            print(f"Warning: Step '{step}' not found in this checklist.")

    def is_done(self, step: str) -> bool:
        """Checks if a specific step is done."""
        return self.state.get(step, False)

    def all_done(self) -> bool:
        """Checks if all steps in the checklist are True."""
        return all(self.state.values())

    def reset(self):
        """Resets all steps in this checklist back to False."""
        for step in self.state:
            self.state[step] = False

class ChatChecklist:
    def __init__(self):
        # Define the steps for each checklist
        phase_steps = [
            "got_name",
            "picked_topic",
            "story_selected",
            "entry_point_phase",
            "engagement_phase",
            "experimental_phase",
            "resolution_phase"
        ]

        sub_phase_steps = [
            "initial_entry_prompt",
            "initial_engagement_prompt",
            "initial_experiment_prompt"
            # Add more sub-phases here as needed
        ]

        # Create instances of ChecklistState to manage each list
        self.phases = ChecklistState(phase_steps)
        self.sub_phases = ChecklistState(sub_phase_steps)

    def reset_all(self):
        """Resets all phases and sub-phases."""
        print("--- CHECKLIST RESET ---")
        self.phases.reset()
        self.sub_phases.reset()