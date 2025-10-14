import re

valid_topics =[
        "Living vs. Non-Living Things", # LT01
        "Basic Needs of Living Things", # LT02
        "Parts of Plants and Animals", # LT03
        "Characteristics of Living Things - Growth", # LT04
        "Characteristics of Living Things - Response", # LT05
        "Characteristics of Living Things - Reproduction", # LT06
        "Characteristics of Living Things - Survival & Extinction" # LT07
    ]

sample_stories = {
    1: "The Adventures of Sparky the Curious Cat",
    2: "Luna and the Magical Forest",
    3: "Tommy's Time-Traveling Telescope"
}

class InputEvaluator:
    # --- NAME EXTRACTION ---
    def extract_name(self, text: str) -> str | None:
        """Extracts a name if the input contains 'my name is' or looks like a name."""
        text = text.strip()

        # Case 1: Match "my name is <name>"
        match = re.search(r"\bmy name is\s+([A-Za-z]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).title()

        # 2. Handle input that seems to be a single name (like "Juan" or "Anna")
        if re.fullmatch(r"[A-Za-z]+", text):
            return text.capitalize()

        # No valid name found
        return None

    def is_empty_name_phrase(self, text: str) -> bool:
        """Checks if the user said 'my name is' but didn't provide a name."""
        text = text.strip().lower()
        return text == "my name is" or text.startswith("my name is ")

     # --- TOPIC EXTRACTION ---
    def extract_topic(self, text: str) -> str | None:
        """Extracts a topic by matching user input with valid topics."""
        text = text.strip().lower()

        # Directly match if user types something like "I want to learn about <topic>"
        match = re.search(r"\b(?:learn about|study)\s+(.+)", text, re.IGNORECASE)
        if match:
            user_topic = match.group(1).strip().lower()
        else:
            user_topic = text

        # Check for partial matches with valid topics
        for topic in valid_topics:
            if any(word in topic.lower() for word in user_topic.split()):
                if any(key in topic.lower() for key in user_topic.split()):
                    return topic

        # Try exact phrase match (case-insensitive)
        for topic in valid_topics:
            if topic.lower() in text:
                return topic

        return None

    def is_empty_topic_phrase(self, text: str) -> bool:
        """Checks if the user said a topic but didn't provide a specific topic."""
        text = text.strip().lower()
        return text == "i want to learn about" or text.startswith("i want to learn about ")
    
    def extract_story_choice(self, text: str) -> int | None:
        """Extracts numeric story choice from input (e.g., '1', 'story 2')."""
        text = text.strip().lower()
        match = re.search(r"\b(\d+)\b", text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None