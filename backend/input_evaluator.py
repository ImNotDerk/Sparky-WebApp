import re

class InputEvaluator:
    def __init__ (self, valid_topics: list[str]):
        """Initializes the evaluator with a dynamic list of valid topics."""
        self.valid_topics = valid_topics

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
        for topic in self.valid_topics:
            if any(word in topic.lower() for word in user_topic.split()):
                if any(key in topic.lower() for key in user_topic.split()):
                    return topic

        # Try exact phrase match (case-insensitive)
        for topic in self.valid_topics:
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

    def is_answer_correct(self, user_input: str, expected_phrases: dict) -> bool:
        """Checks if user input matches any of the expected phrases for phase completion."""
        # Extract keywords and examples safely
        keywords = [kw.lower() for kw in expected_phrases.get("keywords", [])]
        examples = [ex.lower() for ex in expected_phrases.get("examples", [])]

        # Check if any keyword appears in the user input
        if any(kw in user_input for kw in keywords):
            return True

        # Check if user input matches any example closely
        for ex in examples:
            # Allow minor variations (e.g., punctuation or spacing differences)
            if user_input == ex or ex in user_input or user_input in ex:
                return True

        return False