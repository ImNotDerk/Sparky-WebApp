import re
from google import genai
from google.genai import types
from session_data_manager import SessionData

class InputEvaluator:
    def __init__(self, valid_topics: list[str], genai_client, model_uri: str):
        """
        Initializes the evaluator with valid topics and the AI client.

        Args:
            valid_topics (list[str]): A list of valid topic strings.
            genai_client: An instance of the Google GenAI client.
            model_uri (str): The URI of the generative AI model to use.
        """
        self.valid_topics = valid_topics
        self.client = genai_client
        self.model_uri = model_uri

        # Define a base config for all *internal* evaluator AI calls
        self._evaluator_config = types.GenerateContentConfig(
            system_instruction="You are a silent, logical evaluator. Your only job is to analyze the user's input based on the given task and provide the answer in the exact format requested.",
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
            ]
        )

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
        """
        Extracts a topic only if the user's text contains
        the *exact*, full topic name (case-insensitive).
        """
        # 1. Normalize the user's input to lowercase
        user_text_lower = text.strip().lower()

        # 2. Check if any valid topic name is contained in the input
        for topic in self.valid_topics:
            # 3. Find the topic (case-insensitive) in the user's text
            if topic.lower() in user_text_lower:
                # 4. Return the *original, official* topic name
                return topic

        # 5. If no exact match is found
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

    async def is_experiment_valid(self, experiment_idea: str, session_data: SessionData) -> bool:
        """Uses the AI model to validate if the experiment idea is a valid test."""

        # --- 1. Get Context from SessionData ---
        # Get the hypothesis and topic to give the AI context.
        hypothesis = session_data.important_conversation_data.get("last_hypothesis", "their guess")
        topic = session_data.onboarding_data.get("chosen_topic", "the main topic")

        # --- 2. Create a Detailed, Criteria-Based Prompt ---
        prompt_text = (
            f"You are an AI evaluator. Your job is to determine if a child's experiment idea is valid based on specific criteria.\n\n"
            f"--- CONTEXT ---\n"
            f"Learning Topic: \"{topic}\"\n"
            f"Child's Hypothesis (their guess): \"{hypothesis}\"\n\n"

            f"--- CHILD'S EXPERIMENT IDEA ---\n"
            f"\"{experiment_idea}\"\n\n"

            f"--- CRITERIA FOR A VALID EXPERIMENT ---\n"
            f"1.  **Is it a Genuine Attempt?** It must NOT be 'I don't know', 'no', 'you tell me', or a clearly silly/unrelated answer (e.g., 'let's eat pizza').\n"
            f"2.  **Is it Testable?** Does it propose a 'what if' scenario, an action, or a comparison? (e.g., 'put one in the dark', 'see what happens if we add water').\n"
            f"3.  **Is it Relevant?** Does this experiment actually test the hypothesis?\n\n"

            f"--- TASK ---\n"
            f"Analyze the child's idea against the criteria. Is it a valid experiment?\n"
            f"Respond with a single word: **VALID** (if it meets all 3 criteria) or **NOT VALID** (if it fails even one).\n\n"
            f"Response:"
        )

        # --- 3. Call the API ---
        response = await self.client.aio.models.generate_content(
            model=self.model_uri,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt_text)] # Use the formatted prompt
                )
            ],
            # FIX: The parameter is 'generation_config', not 'config'
            config=self._evaluator_config
        )

        print(f"EVALUATOR Response: {response.text}")

        # --- 4. Parse the Response (with fixes) ---
        # FIX: The simple text response is on 'response.text'
        result_text = response.text.strip().upper()

        # Check for YES. Anything else (NO, or garbled text) is treated as invalid.
        return result_text == "VALID"