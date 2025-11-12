import re
from google import genai
from google.genai import types
from session_data_manager import SessionData

class InputEvaluator:
    def __init__(self, stories_data: list[dict], topics_data: list[dict], genai_client, model_uri: str):
        """
        Initializes the evaluator with valid topics and the AI client.

        Args:
            topics_data (list[dict]): Topics data loaded from JSON.
            stories_data (list[dict]): Stories data loaded from JSON.
            genai_client: An instance of the Google GenAI client.
            model_uri (str): The URI of the generative AI model to use.
        """
        self.stories_data = stories_data
        self.topics_data = topics_data
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

    def extract_topic(self, text: str) -> str | None:
        """
        Extracts a topic only if the user's text contains
        the *exact*, full topic name (case-insensitive).
        """
        # 1. Normalize the user's input to lowercase
        user_text_lower = text.strip().lower()

        # 2. Loop through the LIST of DICTIONARIES
        for topics in self.topics_data:

            # 3. Safely get the topic name STRING
            topic_name = topics.get("topic_name")

            # 5. Now, we know topic_name is a string. Run the check.
            if topic_name.lower() in user_text_lower:
                # 6. Return the *original, official* topic name
                return topic_name

        # 7. If no exact match is found
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

    # fix these functions somehow since they are being used in chat_logic_service and input_evaluator
    def _find_story_by_id(self, story_id: str) -> dict | None:
        """
        Helper function to search through the master `self.stories_data` list.
        """
        for story in self.stories_data:
            if story["story_id"] == story_id:
                return story
        return None

    def _get_story_phase_data(self, story_id: str, phase_name: str) -> dict | None:
        """
        Helper function to get phase data from a story.
        """
        story = self._find_story_by_id(story_id)
        if story and "phases" in story and phase_name in story["phases"]:
            return story["phases"][phase_name]
        return None

    def get_topic_details(self, topic_name: str) -> dict | None:
        """
        Returns the details of a specific topic by name.
        Safely handles missing keys and returns None if not found.
        """
        for topic in self.topics_data:
            if topic.get("topic_name") == topic_name:
                return topic
        return None

    async def is_observation_valid(self, observation: str, session_data: SessionData, expected_answers: dict) -> bool:
        """
        Uses the AI model to validate if the observation is valid.
        """

        # --- 1. Get Context ---

        # Get the story OBJECT
        story = session_data.onboarding_data["stories_data"]
        # Get the title string from the object
        story_title = story.get("title")

        # Get the data for the 'entry' phase
        phase_data = self._get_story_phase_data(story["story_id"], "entry")

        entry_story_text = phase_data.get('story')
        main_question = phase_data.get('main_question')
        initial_story_narration = session_data.important_conversation_data["initial_story_narration"]

        # Get the "answer key"
        keywords = [kw.lower() for kw in expected_answers.get("keywords", [])]
        examples = [ex.lower() for ex in expected_answers.get("examples", [])]

        # --- 2. Create the Improved Prompt ---
        prompt_text = (
            f"You are an AI evaluator. Your job is to determine if a child's observation in the first part of a story is valid.\n\n"

            f"--- CONTEXT ---\n"
            f"Topic: {session_data.onboarding_data.get('chosen_topic')}\n"
            f"Story Title: {story_title}\n\n"

            f"THE STORY TEXT THE CHILD READ:\n"
            f"\"\"\"\n"
            f"{entry_story_text}\n"
            f"\"\"\"\n\n"

            f"THE QUESTION YOU ASKED THEM:\n"
            f"\"{main_question}\"\n\n"

            f"--- THE 'ANSWER KEY' (What we are looking for) ---\n"
            f"Expected Keywords: {keywords}\n"
            f"Example Valid Answers: {examples}\n\n"

            f"--- THE CHILD'S ANSWER ---\n"
            f"\"{observation}\"\n\n"

            f"--- CRITERIA FOR A VALID ANSWER ---\n"
            f"1.  **Genuine Attempt:** Is it a real statement? (It must NOT be 'I don't know', 'no', 'you tell me', or gibberish).\n"
            f"2.  **Relevant to the Question:** Does the child's statement *directly and logically answer* 'THE QUESTION YOU ASKED THEM'? (e.g., If the question is 'Why is the plant sad?', 'it needs water' is a VALID answer. If the question is 'What did you see?', 'it needs water' would be NOT VALID).\n"
            f"3.  **Relevant to the Story:** Is the answer *directly based* on 'THE STORY TEXT' provided? (e.g., 'I have a dog' is NOT VALID).\n"
            f"4.  **Matches the Goal:** Does the answer align with the 'Expected Keywords' or 'Example Valid Answers'? It doesn't have to be an exact match, but it must capture the *same idea*.\n\n"

            f"--- TASK ---\n"
            f"Analyze the 'Child's Answer' against all 4 criteria. It is **NOT VALID** if it fails even one.\n"
            f"Respond with a single word: **VALID** or **NOT VALID**.\n\n"
            f"Response:"
    )

        # --- 3. Call the API ---
        response = await self.client.aio.models.generate_content(
            model=self.model_uri,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt_text)]
                )
            ],
            config=self._evaluator_config
        )

        print(f"EVALUATOR Response (Observation): {response.text}")

        # --- 4. Parse the Response ---
        result_text = response.text.strip().upper()
        return result_text == "VALID"

    async def is_hypothesis_valid(self, hypothesis: str, session_data: SessionData, expected_answers: dict) -> bool:
        """
        Uses the AI model to validate if the child's hypothesis is valid
        """
        # Get context of story
        story = session_data.onboarding_data["story_data"]
        story_title = story.get("title")

        # Phase data
        phase_data = self._get_story_phase_data(story["story_id"], "engagement")
        # Engagement phase story text
        engagement_text = phase_data.get("story")

        # Get the "answer key"
        keywords = [kw.lower() for kw in expected_answers.get("keywords", [])]
        examples = [ex.lower() for ex in expected_answers.get("examples", [])]

        hypothesis_question = session_data.important_conversation_data["hypothesis_question"]

        prompt_text = (
            f"You are an AI evaluator. Your job is to determine if a child's hypothesis is valid based on specific criteria.\n\n"
            f"--- CONTEXT ---\n"
            f"Story Title: {story_title}\n"
            f"The Question We Asked: \"{hypothesis_question}\"\n\n"

            f"--- THE 'ANSWER KEY' (What we are looking for) ---\n"
            f"Expected Keywords: {keywords}\n"

            f"Example Valid Hypotheses: {examples}\n\n"

            f"--- THE CHILD'S HYPOTHESIS ---\n"
            f"\"{hypothesis}\"\n\n"

            f"--- CRITERIA FOR A VALID HYPOTHESIS ---\n"
            f"1.  **Genuine Attempt:** Is it a real statement? (It must NOT be 'I don't know', 'no', 'you tell me', or gibberish).\n"
            f"2.  **Relevant Answer:** Does the hypothesis *actually answer* 'The Question We Asked'? (e.g., If the question is 'Why do they need food?', the answer 'They are hungry' is NOT a valid hypothesis, it's just restating the problem).\n"
            f"3.  **Matches the Goal:** Does the hypothesis, *in the child's own words*, capture the *main idea* of the 'Expected Keywords' or 'Example Valid Hypotheses'? (e.g., If the goal is 'energy', a child saying 'to run and play' or 'for energy' is VALID. If the goal is 'growth', 'to grow' is VALID).\n\n"

            f"--- TASK ---\n"
            f"Analyze the 'Child's Hypothesis' against all 3 criteria. It is **NOT VALID** if it fails even one.\n"
            f"Respond with a single word: **VALID** or **NOT VALID**.\n\n"
            f"Response:"
        )

        # --- 3. Call the API ---
        response = await self.client.aio.models.generate_content(
            model=self.model_uri,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt_text)]
                )
            ],
            config=self._evaluator_config
        )

        print(f"EVALUATOR Response (Observation): {response.text}")


    async def is_experiment_valid(self, experiment_idea: str, session_data: SessionData) -> bool:
        """
        Uses the AI model to validate if the experiment idea is a valid test.
        """
        # --- 1. Get Context from SessionData ---
        # Get the hypothesis and topic to give the AI context.
        hypothesis = session_data.important_conversation_data.get("last_hypothesis")
        topic = session_data.onboarding_data.get("chosen_topic")
        topic_details = self.get_topic_details(topic)
        learning_outcome = topic_details.get('learning_outcomes')
        key_concepts = topic_details.get('key_concepts')

        # --- 2. Create a Detailed, Criteria-Based Prompt ---
        prompt_text = (
            f"You are an AI evaluator. Your job is to determine if a child's experiment idea is valid based on specific criteria.\n\n"
            f"--- CONTEXT ---\n"
            f"Child's Hypothesis (their guess): \"{hypothesis}\"\n\n"
            f"--- TOPIC DETAILS ---\n"
            f"Learning Topic: \"{topic}\"\n"
            f"Learning Outcome: {learning_outcome}\n\n"
            f"Key Concepts: {key_concepts}\n\n"

            f"--- CHILD'S EXPERIMENT IDEA ---\n"
            f"\"{experiment_idea}\"\n\n"

            f"--- CRITERIA FOR A VALID EXPERIMENT ---\n"
            f"1.  **Is it a Genuine Attempt?** It must NOT be 'I don't know', 'no', 'you tell me', or a clearly silly/unrelated answer (e.g., 'let's eat pizza').\n"
            f"2.  **Is it Testable?** Does it propose a 'what if' scenario, an action, or a comparison? (e.g., 'put one in the dark', 'see what happens if we add water').\n"
            f"3.  **Is it Relevant?** Does this experiment actually test the hypothesis?\n\n"

            f"--- TASK ---\n"
            f"Analyze the child's idea against the criteria. Is it a valid experiment?\n"
            f"Consider the context provided and the specific details of the experiment idea.\n"
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

        # --- 4. Parse the Response ---
        result_text = response.text.strip().upper()

        # Check for YES. Anything else (NO, or garbled text) is treated as invalid.
        return result_text == "VALID"

    async def is_prediction_valid(self, prediction: str, session_data: SessionData) -> bool:
        """
        Uses the AI model to validate if the prediction is valid.
        """

        # --- 1. Get Context ---
        topic = session_data.onboarding_data.get("chosen_topic", "the main topic")
        topic_details = self.get_topic_details(topic)

        hypothesis = session_data.important_conversation_data.get("last_hypothesis")
        experiment = session_data.important_conversation_data.get("experiment_data")
        learning_outcome = topic_details.get('learning_outcomes')
        key_concepts = topic_details.get('key_concepts')

        # --- 2. Create Improved Prompt ---
        prompt_text = (
            f"You are an AI evaluator. Your job is to determine if a child's prediction for an experiment is valid.\n\n"

            f"--- CONTEXT ---\n"
            f"Learning Topic: \"{topic}\"\n"
            f"Learning Outcome: {learning_outcome}\n\n"
            f"Key Concepts: {key_concepts}\n\n"
            f"Child's Hypothesis: \"{hypothesis}\"\n"
            f"Experiment Being Done: \"{experiment}\"\n\n"

            f"--- CHILD'S PREDICTION ---\n"
            f"\"{prediction}\"\n\n"

            f"--- CRITERIA FOR A VALID PREDICTION ---\n"
            f"1.  **Is it a Concrete Statement?** The prediction must be a real statement about an outcome. It **cannot** be a non-answer ('I don't know', 'no', 'you tell me'), a question, or a vague opinion ('that's weird', 'it will be fun').\n"
            f"2.  **Is it Relevant?** The prediction **MUST** directly answer the question asked by the experiment. (e.g., If the experiment asks 'who will have more *energy*?', the prediction 'I will have a poop' is **NOT VALID** because it does not answer the question about *energy*, even if it is a true statement).\n"
            f"3.  **Is it Plausibly Correct?** Is the prediction scientifically plausible in the context of the experiment and key concepts? (It's okay if it's slightly off, but it must be on the right track).\n\n"

            f"--- TASK ---\n"
            f"Analyze the 'Child's Prediction' against all 3 criteria. It is **NOT VALID** if it fails even one.\n"
            f"Respond with a single word: **VALID** or **NOT VALID**.\n\n"
            f"Response:"
        )

        # --- 3. Call the API ---
        response = await self.client.aio.models.generate_content(
            model=self.model_uri,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt_text)]
                )
            ],
            config=self._evaluator_config
        )

        print(f"EVALUATOR Response (Prediction): {response.text}")

        # --- 4. Parse the Response ---
        result_text = response.text.strip().upper()

        return result_text == "VALID"

    async def is_conclusion_valid(self, conclusion: str, session_data: SessionData) -> bool:
        """
        Uses the AI model to validate if the child's FINAL CONCLUSION is valid.
        This checks if they successfully articulated the main learning point.
        """

        # --- 1. Get Context from SessionData ---
        topic_name = session_data.onboarding_data.get("chosen_topic", "the main topic")

        # Get the "answer key" for the topic
        topic_details = self.get_topic_details(topic_name)
        learning_outcome = topic_details.get('learning_outcomes')
        key_concepts = topic_details.get('key_concepts')

        # Get the history of the inquiry
        hypothesis = session_data.important_conversation_data.get('last_hypothesis')
        experiment = session_data.onboarding_data.get('experiment_data')

        # --- 2. Create a Detailed, Criteria-Based Prompt ---
        prompt_text = (
            f"You are an AI evaluator. Your job is to determine if a child's **final conclusion** (their takeaway) is valid and correct.\n\n"

            f"--- CONTEXT (THE 'ANSWER KEY') ---\n"
            f"Learning Topic: \"{topic_name}\"\n"
            f"The Main Learning Outcome We Want: {learning_outcome}\n"
            f"The Key Concepts: {key_concepts}\n\n"

            f"--- HISTORY OF THE INQUIRY ---\n"
            f"Child's Hypothesis: \"{hypothesis}\"\n"
            f"Experiment Done: \"{experiment}\"\n\n"

            f"--- CHILD'S STATED CONCLUSION ---\n"
            f"We just asked the child what they learned, and they said:\n"
            f"\"{conclusion}\"\n\n"

            f"--- CRITERIA FOR A VALID CONCLUSION ---\n"
            f"1.  **Genuine Attempt:** Is the conclusion a real statement? (It must NOT be 'I don't know', 'no', 'you tell me', etc.).\n"
            f"2.  **Relevant:** Is the conclusion about the experiment, the hypothesis, or the topic? (e.g., It's NOT a random fact like 'I have a dog').\n"
            f"3.  **Captures the Main Lesson:** This is the most important criterion. The child's statement **must** align with the *essence* of the **'Learning Outcome'** or **'Key Concepts'** (the 'Answer Key').\n"
            f"    * It is **NOT VALID** if it *only* states the experiment's result (e.g., 'the plant with food grew') but misses the *bigger lesson* (the 'why', e.g., 'food gives energy/helps us survive').\n"
            f"    * It is **NOT VALID** if it is just a simple definition in response to a question (e.g., 'to keep living' is a definition, not a conclusion from the experiment).\n\n"

            f"--- TASK ---\n"
            f"Analyze the 'Child's Stated Conclusion' against all 3 criteria. It is **NOT VALID** if it's irrelevant, an 'I don't know' response, or if it misses the main lesson.\n"
            f"Respond with a single word: **VALID** or **NOT VALID**.\n\n"
            f"Response:"
        )

        # --- 3. Call the API ---
        response = await self.client.aio.models.generate_content(
            model=self.model_uri,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt_text)]
                )
            ],

            config=self._evaluator_config
        )

        print(f"EVALUATOR Response (Conclusion): {response.text}")

        # --- 4. Parse the Response ---
        result_text = response.text.strip().upper()
        return result_text == "VALID"