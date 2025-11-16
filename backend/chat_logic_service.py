# chat_logic_service.py

"""
This module defines the ChatLogicService, which orchestrates the conversational flow
of the Sparky chatbot. It manages the interaction phases (onboarding, story progression,
experimentation, and resolution) by routing user inputs to appropriate handlers and
generating AI responses. The service is designed to be stateless, relying on the
ChatChecklist to maintain session state.
"""

from tkinter.font import names
from google import genai
from google.genai import types

from checklist_manager import ChatChecklist
from input_evaluator import InputEvaluator
from session_data_manager import SessionData

class ChatLogicService:
    """
    The ChatLogicService class is responsible for the core conversational logic
    of the Sparky chatbot. It acts as a central router, directing user messages
    to specific handler functions based on the current phase of the conversation.
    It integrates with GenAI for AI responses, InputEvaluator for user input parsing,
    and ChatChecklist for state management.

    == Conversational Phase Lifecycle ==

    The chatbot's interaction with the user progresses through a series of distinct phases,
    managed by the `ChatChecklist` and driven by `PHASE_ADVANCE` tokens in the AI's responses.

    1.  **Onboarding (Static Phases):**
        * **`got_name`**: The initial stage where the chatbot asks for and extracts the child's name.
        * **`picked_topic`**: After getting the name, the chatbot asks for a learning topic and then presents a list of stories related to that topic.
        * **`story_selected`**: The child selects a story number, and the chatbot loads the full story data.
        * **Transition:** These phases are handled by direct function calls (`_handle_get_name`, `_handle_pick_topic`, `_handle_select_story`) and explicitly mark steps as `done` in the `checklist`. Once `story_selected` is done, the chatbot transitions to the first dynamic phase: `entry`.

    2.  **Story-Based Learning (Dynamic AI Phases):** These phases involve deeper AI interaction and pedagogical guidance.

        * **`entry` (Observation):**
            * **Purpose:** Introduce the story and prompt the child to make an initial observation about the scene.
            * **Trigger:** Automatically set after `story_selected` is complete.
            * **AI Action:** Narrates the story's introductory scene and asks an open-ended "What do you notice?" question.
            * **Transition:** Adds `[PHASE_ADVANCE:engagement]` to move to the next phase after the narration.

        * **`engagement` (Hypothesis):**
            * **Purpose:** Acknowledge the child's observation and guide them to form a hypothesis (a "guess") about *why* something is happening in the story.
            * **Trigger:** Activated when the child responds to the `entry` phase's observation question.
            * **AI Action:** Praises the observation and asks a "Why do you think...?" or "What's your guess...?" question.
            * **Transition:** Adds `[PHASE_ADVANCE:experiment]` to progress to the experimentation phase.

        * **`experiment` (Experimentation):**
            * **Purpose:** Dynamically create a simple, two-part "thought experiment" directly related to the current `topic` and the child's `hypothesis` to test their idea. Then, ask for a prediction.
            * **Trigger:** Activated when the child provides their hypothesis in the `engagement` phase.
            * **AI Action:** Praises the hypothesis, proposes a "what if" scenario (e.g., "Imagine two Dodos, one with a home, one without..."), and asks "What do you *think* will happen?".
            * **Transition:** Adds `[PHASE_ADVANCE:resolution]` to move to the resolution phase, where the prediction will be evaluated.

        * **`resolution` (Conclusion & Scaffolding):**
            * **Purpose:** Evaluate the child's prediction from the experiment. If correct, state the scientific conclusion. If incorrect or "I don't know," provide hints and re-ask the prediction question until they grasp the concept.
            * **Trigger:** Activated when the child provides their prediction for the experiment.
            * **AI Action (Correct Prediction):** Praises the prediction, articulates the scientific takeaway linking hypothesis, experiment, and topic, tells the final story part, and asks a real-life application question.
            * **AI Action (Incorrect/Uncertain Prediction):** Acknowledges gently, provides a **scaffold/hint** specific to the experiment, and **re-asks the prediction question**.
            * **Transition:** Adds `[PHASE_ADVANCE:completed]` *only if* the child makes a reasonable prediction and understands the conclusion. If not, no token is added, keeping the conversation in `resolution` to offer more guidance.

        * **`completed` (Free-form Chat):**
            * **Purpose:** Allow open-ended conversation after the story's learning objectives have been met.
            * **Trigger:** Activated when the `resolution` phase successfully concludes and issues `[PHASE_ADVANCE:completed]`.
            * **AI Action:** Engages in general, friendly conversation based on the user's input, without specific learning goals.
            * **Transition:** This is the final phase for a given story; there are no further `PHASE_ADVANCE` tokens.
    """
    def __init__(self, genai_client, model_uri: str, input_evaluator: InputEvaluator, stories_data: list[dict], topics_data: list[dict]):
        """
        Initializes the ChatLogicService with necessary dependencies.

        Args:
            genai_client: An instance of the Google GenAI client for API calls.
            model_uri (str): The URI of the generative AI model to use.
            input_evaluator (InputEvaluator): An instance to parse and evaluate user inputs.
            stories_data (list[dict]): A list of dictionaries containing all available story data,
                                     including phases and their content.
            topics_data (list[dict]): A list of dictionaries containing all available topic data.
        """
        self.client = genai_client
        self.model_uri = model_uri
        self.evaluator = input_evaluator
        self.stories_data = stories_data
        self.topics_data = topics_data

    # --- 1. The Main Router ---

    async def process_message(
            self, checklist: ChatChecklist, session_data: SessionData, history: list[types.Content], user_prompt: str
        ) -> tuple[str, ChatChecklist]:
        """
        The main entry point for processing any user message.
        It orchestrates the conversational flow by:
        1. Checking for static onboarding steps (name, topic, story selection).
        2. If onboarding is complete, routing the message to the appropriate
           dynamic story phase handler (`entry`, `engagement`, `experiment`, `resolution`, `completed`).
        3. Extracting and applying any phase advance tokens from the AI's response.

        Args:
            checklist (ChatChecklist): The current state checklist for the conversation.
            history (list[types.Content]): The chat history, including previous user and AI turns.
            user_prompt (str): The latest message from the user.

        Returns:
            tuple[str, ChatChecklist]: A tuple containing the AI's response message
                                      and the updated ChatChecklist.
        """

        current_phase = checklist.phases.get_current_phase()
        bot_reply = None
        choices_to_send = None

        # === A. Handle Static Onboarding ===
        # These are initial, fixed steps to gather basic information from the user.
        if current_phase == "got_name":
            bot_reply = self._handle_get_name(checklist, session_data, user_prompt)
        elif current_phase == "picked_topic":
            bot_reply = self._handle_pick_topic(checklist, session_data, user_prompt)
        elif current_phase == "story_selected":
            bot_reply = await self._handle_select_story(checklist, session_data, user_prompt)

        # === B. Handle Dynamic Story Phases ===
        # Once onboarding is complete, the conversation enters dynamic story phases.
        elif current_phase == "entry_point_phase":
            bot_reply = await self._handle_phase_entry_point(checklist, session_data, history, user_prompt)
        elif current_phase == "engagement_phase":
            bot_reply = await self._handle_phase_engagement(checklist, session_data, history, user_prompt)
        elif current_phase == "experimental_phase":
            bot_reply = await self._handle_phase_experiment(checklist, session_data, history, user_prompt)
        elif current_phase == "conclusion_phase":
            bot_reply = await self._handle_conclusion_phase(checklist, session_data, history, user_prompt)
        elif current_phase == "resolution_phase":
            bot_reply = await self._handle_phase_resolution(checklist, session_data, history, user_prompt)
        elif current_phase == "completed_phase":
            bot_reply = await self._handle_phase_completed(checklist, session_data, history, user_prompt)
        else:
            bot_reply = await self._handle_choice_phase(checklist, session_data, history, user_prompt)

        # --- C. Final Processing ---
        # If no handler produced a reply, provide a fallback message.
        if bot_reply is None:
            bot_reply  = "I'm not sure how to respond. Please try again or reset our chat."

        if current_phase == "got_name":
            choices_to_send = self.get_topic_list()
        elif current_phase == "picked_topic":
            choices_to_send = session_data.onboarding_data.get("topic_stories_list")
        elif current_phase == "pick_new_topic":
            choices_to_send = self.get_topic_list()
            checklist.new_topic()
            return bot_reply, choices_to_send, checklist, session_data
        elif current_phase == "pick_new_story":
            choices_to_send = session_data.onboarding_data.get("topic_stories_list")
            checklist.new_story()
            return bot_reply, choices_to_send, checklist, session_data
        else:
            choices_to_send = {}

        print("current phase:", checklist.phases.get_current_phase())

        return bot_reply, choices_to_send, checklist, session_data

    # --- 2. Static Onboarding Handlers ---

    def _handle_get_name(self, checklist: ChatChecklist, session_data: SessionData, prompt: str) -> str:
        """
        Handles the 'got_name' onboarding step.
        Attempts to extract the child's name from the prompt and stores it.
        If successful, it moves to the next step; otherwise, it re-prompts for the name.
        """
        name = self.evaluator.extract_name(prompt)
        if name:
            session_data.onboarding_data["name"] = name
            checklist.phases.mark_done("got_name") # Mark this phase as done
            return (
                    f"Nice to meet you, {name}! What would you like to learn about today?\n\n"
                   f"Here are the topics you can choose from:\n"
            )
        return "Before we start, can you please tell me your name?"

    def _handle_pick_topic(self, checklist: ChatChecklist, session_data: SessionData, prompt: str) -> str:
        """
        Handles the 'picked_topic' onboarding step.
        Extracts the topic from the user's prompt, stores it, and then presents
        a list of available stories for that topic.
        If the topic is not recognized, it re-prompts.
        """
        topic = prompt
        if topic:
            session_data.onboarding_data["chosen_topic"] = topic
            session_data.onboarding_data["topic_details"] = self.get_topic_details(topic)

            # Filter stories based on the chosen topic and prepare them for display.
            topic_stories = [story for story in self.stories_data if story["topic"] == topic]

            session_data.onboarding_data["topic_stories"] = topic_stories # Save topic stories in session_data
            story_display_list = []
            for story in topic_stories:
                story_display_list.append(f"{story['title']}")

            session_data.onboarding_data["topic_stories_list"] = story_display_list # Store map for selection

            checklist.phases.mark_done("picked_topic")  # Mark this phase as done

            return (
                f"Great choice! We're going to learn about **{topic}**.\n\n"
                f"Here are the stories you can choose from:\n"
            )
        return "I didn't quite get that. What topic would you like to learn about today?"

    async def _handle_select_story(self, checklist: ChatChecklist, session_data: SessionData, prompt: str) -> str:
        """
        Handles the 'story_selected' onboarding step.
        Extracts the story choice (number) from the user, retrieves the corresponding
        story data, and sets the initial dynamic phase to 'entry'.
        If the choice is invalid, it re-prompts.
        """
        story_map = session_data.onboarding_data.get("topic_stories")

        for story in story_map:
            # Check if the title matches
            if story.get("title") == prompt:
                session_data.onboarding_data["story_data"] = story
                checklist.phases.mark_done("story_selected") # Mark this phase as done
                return await self._handle_phase_entry_point(checklist, session_data, None, prompt)
            else:
                session_data.onboarding_data["story_data"] = {}

        return "I didn't quite get that. What story would you like to learn about today?"

    # --- 3. Dynamic AI Phase Handlers ---
    # These functions manage the core learning experience, guiding the child
    # through observation, hypothesis, experiment, and conclusion.

    async def _handle_phase_entry_point(self, checklist: ChatChecklist, session_data: SessionData, history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Entry Point' phase.
        SPARKY narrates the initial scene of the story and asks an open-ended
        question to encourage the child to make an observation.
        This phase transitions to 'engagement'.
        """
        story = session_data.onboarding_data.get("story_data") # Get Data for the chosen story
        phase_data = self._get_story_phase_data(story["story_id"], "entry") # Get Data for Entry Phase
        name = session_data.onboarding_data.get('name') # Get child name

        # Get "answer key" from keywords and examples in stories kb
        keywords = ', '.join(phase_data.get('expected_answer', {}).get('keywords', []))
        examples = ', '.join(phase_data.get('expected_answer', {}).get('examples', []))

        # Get entry point story data
        entry_point_story = phase_data.get('story')
        main_question = phase_data.get('main_question')

        entry_prompt_done = checklist.sub_phases.is_done("initial_entry_prompt")

        if not entry_prompt_done:
            wrapped_prompt = ( # First time backend interacting with API
                f"Start the story '{story['title']}'.\n"
                f"Your goal is to get the child to make an observation.\n\n"
                f"TASK:\n"
                f"1. Narrate this scene in your own simple, fun, first-person voice if you see the word \"SPARKY\":\n"
                f"   \"\"\"\n"
                f"   {phase_data['story']}\n"
                f"   \"\"\"\n"
                f"2. After narrating, ask an open-ended **observation question** that will entice the child to think more about the story.\n"
                f"-  Possibly make it similar to this: {main_question}"
            )

            # Store observation question to pass to evaluator
            bot_reply = await self._call_ai(session_data, history, wrapped_prompt)
            session_data.important_conversation_data["initial_story_narration"] = bot_reply

            checklist.sub_phases.mark_done("initial_entry_prompt")
            return bot_reply

        is_observation_valid = await self.evaluator.is_observation_valid(user_prompt, session_data, phase_data["expected_answer"])

        if is_observation_valid: # if the child got the observation right, pass their observation to the next phase
            checklist.phases.mark_done("entry_point_phase") # Mark this phase as done
            checklist.sub_phases.mark_undone("initial_entry_prompt")
            return await self._handle_phase_engagement(checklist, session_data, history, user_prompt)

        else:
            print("Observation incorrect, scaffolding.")
            wrapped_prompt = (
                f"You are a patient and friendly learning guide. Your goal is to help a child named {name} think through a science question about a story.\n\n"

                f"--- STORY SO FAR ---\n"
                f"\"{entry_point_story}\"\n\n"

                f"--- THE QUESTION YOU ASKED ---\n"
                f"\"{main_question}\"\n\n"

                f"--- THE CHILD'S ANSWER ---\n"
                f"\"{user_prompt}\"\n\n"

                f"--- WHAT A CORRECT ANSWER LOOKS LIKE ---\n"
                f"A correct answer would be about these concepts: {keywords}\n"
                f"Good examples would be: {examples}\n\n"

                f"--- YOUR TASK: RESPOND TO THE CHILD ---\n"
                f"The child's answer wasn't right. Analyze their answer and choose ONE of the following paths. Write *only* the response to the child.\n\n"

                f"PATH 1: The answer is a genuine try, but conceptually wrong.\n"
                f"(e.g., They said 'the rock' when the answer is 'the bee'; they guessed a related, but incorrect, idea).\n"
                f"1.  **Acknowledge & Validate:** Start with a positive, encouraging phrase (e.g., 'That's a really sharp observation!', 'Ooh, that's a close one! I see why you said that.').\n"
                f"2.  **Provide a Scaffolding Hint:** Gently guide them. *Do not give the answer.* Point their attention back to a *key detail* in the story they might have missed or a key word in the question.\n"
                f"    * *Hint Example:* If the story is '...a buzzing bee, a tall sunflower, a smooth gray rock...' and they said 'the rock', a perfect hint is: 'You're right, Sparky *did* see a rock! But remember, the question is about *living* things. Which of those things in the garden seemed to be moving or growing all by itself?'\n"
                f"3.  **Re-ask the Question:** Re-phrase the question slightly to help them focus (e.g., 'So, which one do you think is alive?').\n\n"

                f"PATH 2: The answer is off-topic, a side-track, or 'I don't know'.\n"
                f"(e.g., 'pizza', 'i want to play', 'idk', 'you tell me', 'i'm bored').\n"
                f"1.  **Gently Re-focus:** Be patient, warm, and bring them back. Don't scold. (e.g., 'Hehe, pizza sounds yummy! But let's help Sparky finish his adventure first!', or 'That's totally okay! We can figure it out together, {session_data.onboarding_data.get('student_name', 'friend')}.')\n"
                f"2.  **Simplify & Remind:** Briefly repeat the *most important* part of the story or question in one simple sentence.\n"
                f"    * *Example:* 'Remember, Sparky saw four things in the garden. He was wondering which ones were alive.'\n"
                f"3.  **Re-ask the Question:** Ask the main question again, simply and clearly. (e.g., 'Can you tell me one of the things he saw that you think is living?').\n\n"

                f"--- STYLE ---\n"
                f"Be warm, curious, and encouraging. Use simple, grade-3-level language. Use the child's name, {name}, once if it feels natural."
            )

        return await self._call_ai(session_data, history, wrapped_prompt)

    async def _handle_phase_engagement(self, checklist: ChatChecklist, session_data: SessionData,history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Engagement' (Hypothesis) phase.
        This function is now conditional. It first asks for a hypothesis,
        then evaluates the child's answer. If the hypothesis is correct,
        it manually calls the next phase. If not, it scaffolds.
        """

        story = session_data.onboarding_data["story_data"] # Get Data for the chosen story

        # Get data for engagement phase
        phase_data = self._get_story_phase_data(story["story_id"], "engagement")

        # Get "answer key" from keywords and examples in stories kb
        keywords = ', '.join(phase_data.get('expected_answer', {}).get('keywords', []))
        examples = ', '.join(phase_data.get('expected_answer', {}).get('examples', []))

        # Check if we've already asked for the hypothesis.
        engagement_prompt_done = checklist.sub_phases.is_done("initial_engagement_prompt")

        if not engagement_prompt_done:
            # This is the FIRST call to this function.
            # The 'user_prompt' is the CORRECT OBSERVATION from the 'entry' phase.
            user_observation = user_prompt

            # Save the observation for context
            session_data.important_conversation_data["last_observation"] = user_observation

            # This is the "starter" prompt for the engagement phase.
            wrapped_prompt = (
                f"The user just made a correct observation: \"{user_observation}\"\n"
                f"TASK: Your job is to ask for their hypothesis (their 'guess').\n"
                f"1. Acknowledge their observation.\n"
                f"2. Ask a simple, Socratic 'why' question to get their **hypothesis**, Something like: \"{phase_data['main_question']}\").\n"
            )

            checklist.sub_phases.mark_done("initial_engagement_prompt") # Set initial engagement prompt to done
            bot_reply = await self._call_ai(session_data, history, wrapped_prompt)
            session_data.important_conversation_data["hypothesis_question"] = bot_reply

            return bot_reply

        # If the flag is True, this is a SUBSEQUENT call.
        # The 'user_prompt' is now the child's HYPOTHESIS.
        else:
            user_hypothesis = user_prompt

            is_hypothesis_valid = await self.evaluator.is_hypothesis_valid(user_hypothesis, session_data, phase_data["expected_answer"])
            print(is_hypothesis_valid)

            if is_hypothesis_valid:
                # The hypothesis is correct!
                # 1. Reset the flag for the next time we run this story
                checklist.phases.mark_done("engagement_phase") # Mark this phase as done
                checklist.sub_phases.mark_undone("initial_engagement_prompt") # Mark initial engagement prompt as undone
                # 2. Manually call the NEXT phase
                return await self._handle_phase_experiment(checklist, session_data, history, user_hypothesis)

            else:
                # The hypothesis is incorrect, "I don't know," or silly.
                # We scaffold and stay in this phase.
                print("Hypothesis incorrect, scaffolding.")
                wrapped_prompt = (
                    f"The child tried to answer the hypothesis question but was incorrect!\n"
                    f"Their previous guess was: \"{user_prompt}\"\n"
                    f"The current topic is: \"{session_data.onboarding_data['chosen_topic']}\"\n"
                    f"Story reminder:\n\"{phase_data['story']}\"\n\n"
                    f"TASK:\n"
                    f"1. Encourage them to think again kindly (e.g., \"That's an interesting guess! Let's think about it...\").\n"
                    f"2. Give a hint related to the topic and the story scene.\n"
                    f"3. Re-ask the hypothesis question in a simpler way: {phase_data['main_question']}\n"
                    f"4. The hypothesis should be similar to these expected answers:\n"
                    f"- Expected Keywords: {keywords}"
                    f"- Examples: {examples}"
                )

                return await self._call_ai(session_data, history, wrapped_prompt)

    async def _handle_phase_experiment(self, checklist: ChatChecklist, session_data: SessionData, history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Experiment' phase in multiple turns:
        1.  (Turn 1) Receives a HYPOTHESIS, asks for an experiment idea.
        2.  (Turn 2) Receives an EXPERIMENT IDEA, evaluates it.
            a. If valid, accepts it and asks for a PREDICTION.
            b. If invalid, proposes its own experiment and asks for a PREDICTION.
        3.  (Turn 3+) Receives a PREDICTION, evaluates it.
            a. If valid, moves to the conclusion phase.
            b. If invalid, scaffolds and re-asks for a prediction, looping this step.
        """

        # --- 1. Get Context and Data ---

        # Per your request, using 'story_data'.
        # Note: This line will crash if 'story_data' isn't a real key.
        # Make sure your SessionData __init__ has: "story_data": {}
        story = session_data.onboarding_data["story_data"]
        topic = session_data.onboarding_data["chosen_topic"]

        # --- 2. Check Sub-Phase Flags ---
        # We use two flags to know which turn we're on.

        # Flag 1: Have we already asked the user for their experiment idea?
        asked_for_experiment = checklist.sub_phases.is_done("initial_experiment_prompt")

        # Flag 2: Have we already asked the user for their prediction?
        asked_for_prediction = checklist.sub_phases.is_done("asked_for_prediction_prompt")

        # --- 3. Main Logic: A 3-Path 'if/elif/else' ---

        # --- [PATH 1: This is TURN 1 (Input: Hypothesis)] ---
        # If we haven't asked for an experiment yet, this is the first time we're in this phase.
        if not asked_for_experiment:
            # The 'user_prompt' is the child's (presumed valid) HYPOTHESIS from the previous phase.
            user_hypothesis = user_prompt
            # Save the hypothesis so we can use it in later turns.
            session_data.important_conversation_data["last_hypothesis"] = user_hypothesis

            # This prompt praises the hypothesis and asks for an experiment idea.
            wrapped_prompt = (
                f"The user's hypothesis about our topic ('{topic}') is: \"{user_hypothesis}\"\n\n"
                f"--- TASK: Ask for Child's Experiment ---\n"
                f"1.  **Acknowledge the Hypothesis:** Praise their guess (e.g., \"Wow, '{user_hypothesis}'... that's a smart hypothesis! 🧑‍🔬\").\n"
                f"2.  **Ask for Experiment:** Now, ask the child if *they* can think of a simple 'what if' experiment to test their idea. (e.g., \"How could we test that? Can you think of a 'what if' experiment for us to try?\")\n"
            )

            # Set Flag 1 to TRUE so we don't run this block again.
            checklist.sub_phases.mark_done("initial_experiment_prompt")

            # Return the AI's question. We stay in 'experimental_phase' for the next turn.
            return await self._call_ai(session_data, history, wrapped_prompt)

        # --- [PATH 2: This is TURN 2 (Input: Experiment Idea)] ---
        # If we HAVE asked for an experiment, but HAVE NOT asked for a prediction yet.
        elif not asked_for_prediction:
            # The 'user_prompt' is the child's EXPERIMENT IDEA.
            user_experiment_idea = user_prompt

            # We evaluate the user's experiment idea (e.g., is it "I don't know" or "let's eat pizza"?).
            is_valid_experiment = await self.evaluator.is_experiment_valid(user_experiment_idea, session_data)

            # --- [PATH 2A: The experiment idea is VALID] ---
            if is_valid_experiment:
                # The child's idea is good! We praise it and ask for their prediction.
                wrapped_prompt = (
                    f"The child proposed a valid experiment idea: \"{user_experiment_idea}\"\n\n"
                    f"--- TASK: Ask for Prediction ---\n"
                    f"1.  **Praise the Experiment Idea:** (e.g., \"That's a fantastic experiment! Let's see what happens!\").\n"
                    f"2.  **Restate the Experiment Clearly:** Make sure to clearly restate their experiment idea.\n"
                    f"3.  **Ask for Prediction:** Now, ask them: \"What do you *think* will happen in your experiment?\"\n"
                )

                # Set Flag 2 to TRUE. This is the fix for your main logic bug.
                # We now know we have a valid experiment AND we've asked for a prediction.
                checklist.sub_phases.mark_done("asked_for_prediction_prompt")

                # We stay in 'experimental_phase' to wait for the prediction.
                bot_reply = await self._call_ai(session_data, history, wrapped_prompt)

                session_data.important_conversation_data["experiment_data"]

                return bot_reply

            # --- [PATH 2B: The experiment idea is NOT VALID] ---
            else:
                # The child's idea was bad (or "I don't know"). We propose our own.
                user_hypothesis = session_data.important_conversation_data.get("last_hypothesis", "No hypothesis provided")

                # This prompt proposes the AI's own experiment.
                wrapped_prompt = (
                    f"--- CONTEXT ---\n"
                    f"The user's hypothesis is: \"{user_hypothesis}\".\n"
                    f"Their experiment idea was: \"{user_experiment_idea}\" (they are stuck or said 'I don't know').\n"
                    f"Current topic: \"{topic}\".\n\n"

                    f"--- CRITICAL RULES FOR EXPERIMENT CREATION ---\n"
                    f"1.  **BE DIRECT, NO CONFUSING ANALOGIES:** The experiment you propose **MUST** be a simple, direct 'what if' scenario. Do **NOT** use complex metaphors or analogies about unrelated things (e.g., for a 'food/energy' hypothesis, do not bring up cars/fuel or batteries). This is confusing for a child.\n"
                    f"2.  **STAY FOCUSED ON THE TOPIC:** The experiment **MUST** be a direct test of the hypothesis, using simple concepts from the **'{topic}'** itself.\n\n"

                    f"--- EXAMPLES OF GOOD, SIMPLE EXPERIMENTS ---\n"
                    f"Here are examples of simple, direct experiments you have used before:\n"
                    f"-   (Hypothesis: 'food gives energy'): 'What if you eat a big lunch, but I don't eat anything? Who will have more energy?'\n"
                    f"-   (Hypothesis: 'food helps us grow'): 'What if we have two plants, but only give one plant food? Which one will grow?'\n\n"

                    f"--- YOUR TASK ---\n"
                    f"1.  **Start with a gentle, encouraging phrase** (e.g., 'No worries at all! That's what I'm here for. I have an idea!').\n"
                    f"2.  **Propose a simple 'what if' experiment** that follows the **CRITICAL RULES** and is inspired by the **GOOD EXPERIMENT EXAMPLES**.\n"
                    f"3.  **Ensure** it clearly tests the user's hypothesis: \"{user_hypothesis}\".\n"
                    f"4.  **End by asking for their prediction** (e.g., \"What do you predict will happen?\").\n"
                )

                # We also set Flag 2 to TRUE here.
                # This tells the next turn (Path 3) to evaluate the user's prediction
                # for the AI'S experiment.
                checklist.sub_phases.mark_done("asked_for_prediction_prompt")

                # We stay in 'experimental_phase' to wait for the prediction.
                bot_reply = await self._call_ai(session_data, history, wrapped_prompt)
                session_data.important_conversation_data["experiment_data"] = bot_reply

                return bot_reply

        # --- [PATH 3: This is TURN 3+ (Input: Prediction)] ---
        # If Flag 1 is TRUE and Flag 2 is TRUE, we are here.
        # The 'user_prompt' is the child's PREDICTION.
        else:
            # We evaluate the prediction.
            is_prediction_valid = await self.evaluator.is_prediction_valid(user_prompt, session_data)

            # --- [PATH 3A: The prediction is VALID] ---
            if is_prediction_valid:
                # The phase is finally complete!
                checklist.phases.mark_done("experimental_phase")

                # Reset all sub-phase flags for a clean slate next time.
                checklist.sub_phases.mark_undone("initial_experiment_prompt")
                # We reset the flag we just used.
                checklist.sub_phases.mark_undone("asked_for_prediction_prompt")

                # Manually call the *next* phase handler, passing in the valid prediction.
                return await self._handle_conclusion_phase(checklist, session_data, history, user_prompt)

            # --- [PATH 3B: The prediction is NOT VALID] ---
            else:
                # The prediction is wrong.
                # We must scaffold and re-ask the prediction question.
                user_prediction = user_prompt
                wrapped_prompt = (
                    f"The child's prediction was: \"{user_prediction}.\"\n"
                    f"Try to lead it to a valid prediction but don't give it the exact answer."
                    f"Ask for their prediction again."
                    # This prompt should be improved, but this is the logic.
                )

                # This keeps the user in a loop. Their next message will come
                # right back here (Path 3) to be evaluated again.
                return await self._call_ai(session_data, history, wrapped_prompt)

    async def _handle_conclusion_phase(self, checklist: ChatChecklist, session_data: SessionData, history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Conclusion' (or Resolution) phase.
        TURN 1: Praises the user's correct prediction and ASKS for the final conclusion.
        TURN 2: Receives the conclusion, VALIDATES it, and scaffolds if necessary.
        """
        # --- 1. Get Context and Data ---
        story = session_data.onboarding_data["story_data"]
        topic = session_data.onboarding_data["chosen_topic"]
        topic_details = self.get_topic_details(topic)

        # Get context from the session
        hypothesis = session_data.important_conversation_data.get('last_hypothesis', 'their guess')
        experiment = session_data.important_conversation_data.get('experiment_data', 'our experiment')
        learning_outcome = topic_details.get('learning_outcomes')
        key_concepts = topic_details.get('key_concepts')

        # --- 2. Check: Is this Turn 1 or Turn 2? ---

        initial_prompt_done = checklist.sub_phases.is_done("initial_conclusion_prompt")

        # --- [PATH 1: This is TURN 1 (Input: Valid Prediction)] ---
        # If we have NOT asked for a conclusion yet.
        if not initial_prompt_done:
            # The 'user_prompt' is the child's VALID PREDICTION from the 'experiment' phase.
            user_prediction = user_prompt

            # [IMPROVED]: This prompt praises the prediction, states the result,
            # and asks for the "so what?" (the conclusion).
            wrapped_prompt = (
                f"--- CONTEXT ---\n"
                f"The user's hypothesis was: \"{hypothesis}\"\n"
                f"The experiment was: \"{experiment}\"\n"
                f"The user just correctly predicted the outcome: \"{user_prediction}\"\n"
                f"The topic is: \"{topic}\"\n\n"
                f"--- TASK: Praise Prediction and Ask for Conclusion ---\n"
                f"1.  **Praise their prediction:** Start by confirming they were right!\n"
                f"2.  **State the Outcome:** Briefly state the result of the experiment.\n"
                f"3.  **Ask for the 'Why' / Conclusion:** Ask them to form the final conclusion. This is the most important step. Ask *what this teaches us* about the main topic. (e.g., \"So, what does this whole experiment tell us about '{topic}'?\", \"What did you learn from this?\").\n"
            )

            # Set the flag so we know this step is done.
            checklist.sub_phases.mark_done("initial_conclusion_prompt")

            # We stay in 'conclusion_phase' to wait for their answer.
            return await self._call_ai(session_data, history, wrapped_prompt)

        # --- [PATH 2: This is TURN 2+ (Input: Child's Conclusion)] ---
        # If we have already asked for the conclusion, we're here.
        # The 'user_prompt' is the child's ATTEMPT at a conclusion.
        else:
            # We now use your new validator to check their answer.
            is_conclusion_valid = await self.evaluator.is_conclusion_valid(user_prompt, session_data)

            # --- [PATH 2A: The Conclusion is VALID] ---
            if is_conclusion_valid:
                # The child got it! The learning loop is complete.
                checklist.phases.mark_done("conclusion_phase")

                # Reset the sub-phase flag for next time
                checklist.sub_phases.mark_undone("asked_for_prediction_prompt")
                checklist.sub_phases.mark_undone("initial_conclusion_prompt")

                return await self._handle_phase_resolution(checklist, session_data, history, user_prompt)

            # --- [PATH 2B: The Conclusion is NOT VALID] ---
            else:
                # The child's conclusion was wrong, "I don't know," or missed the point.
                # We need to scaffold and re-ask.

                wrapped_prompt = (
                    f"--- CONTEXT ---\n"
                    f"The child's stated conclusion was: \"{user_prompt}\".\n"
                    f"**CRITICAL: Our validator has confirmed this conclusion is NOT VALID.**\n"
                    f"This means the child's answer is one of these things:\n"
                    f"  a) An 'I don't know' or 'no' response.\n"
                    f"  b) An incorrect statement.\n"
                    f"  c) A true but **irrelevant fact** (a 'distractor'). (e.g., If the lesson is 'food gives energy', a distractor is 'food makes poop'. Both are true, but only one is the *lesson*).\n"

                    f"\n--- THE 'ANSWER KEY' (What we are guiding them toward) ---\n"
                    f"The Topic: \"{topic}\".\n"
                    f"The Hypothesis: \"{hypothesis}\".\n"
                    f"The Experiment: \"{experiment}\".\n"
                    f"The Main Lesson: \"{learning_outcome}\" (related to \"{key_concepts}\").\n\n"

                    f"--- YOUR TASK: Guide the Child to the Correct Conclusion ---\n"
                    f"**Rule 1: DO NOT AGREE with their incorrect conclusion.** Never say 'You're right' or 'Yes!' if their answer is NOT VALID. This is the most important rule.\n"
                    f"**Rule 2: DO NOT GIVE THE ANSWER.** Do not just state the main lesson. You must guide them to say it themselves.\n"
                    f"**Rule 3: You MUST scaffold and re-ask.**\n\n"

                    f"Follow these steps:\n"
                    f"1.  **Gently Acknowledge (but do not agree):** Start with a neutral, encouraging phrase. (e.g., \"That's an interesting thought!\", \"I see what you're thinking...\", \"Let's think about that...\").\n"
                    f"2.  **Redirect and Hint:** If their answer was a 'distractor', acknowledge it briefly but pivot back to the *main lesson*. (e.g., \"That's true, but what about the *energy* we talked about?\"). Give a strong hint that connects the **experiment's result** back to the **hypothesis**.\n"
                    f"3.  **Re-ask the Question:** Ask the 'what did you learn' question again in a simple, guiding way. (e.g., \"So, what's the big lesson our experiment teaches us about {topic}?\").\n"
                )

                # We stay in 'conclusion_phase' by NOT marking it done.
                # This keeps the user in this loop until the validator passes.
                return await self._call_ai(session_data, history, wrapped_prompt)

    async def _handle_phase_resolution(self, checklist: ChatChecklist, session_data: SessionData, history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Resolution' (final) phase.
        This function now ASSUMES the child's conclusion is valid.
        It praises their conclusion, tells the final story, and asks the real-life question.
        """

        # --- 1. Get Context and Data ---
        story = session_data.onboarding_data["story_data"]
        topic = session_data.onboarding_data["chosen_topic"]

        # This 'phase_data' contains the final story part and the real-life question
        phase_data = self._get_story_phase_data(story["story_id"], "resolution")

        # The 'user_prompt' in this phase is the child's CONCLUSION.
        user_conclusion = user_prompt

        # --- 2. Create the "Finale" Prompt ---
        # No validation is performed. We immediately go to the wrap-up.

        wrapped_prompt = (
            f"The user's final conclusion was: \"{user_conclusion}\".\n"
            f"The current topic is: \"{topic}\".\n\n"

            f"--- TASK: Praise, Tell Final Story, and Apply to Real Life ---\n"
            f"**This is the final step. Acknowledge their conclusion and wrap up.**\n\n"

            f"1.  **Praise their Conclusion:** Start with a positive phrase that praises their effort and repeats their idea. (e.g., \"That's a super smart way to say it! You figured out that {user_conclusion}!\").\n"
            f"2.  **Tell the Resolution Story:** Now, smoothly transition but tell them you're going back to the story so they won't get confused and tell the final story part, which reinforces this lesson: \"{phase_data['story']}\"\n"
            f"3.  **Ask the Final Question (Real-Life Connection):** Ask the real-life application question from the story data. This connects the lesson to their life. (e.g., \"{phase_data['main_question']}\").\n"
        )

        # --- 3. Mark Phase as Done ---
        # This is the end of the learning loop for this story.
        checklist.phases.mark_done("resolution_phase")

        # Call the AI and return its final response
        return await self._call_ai(session_data, history, wrapped_prompt)


    async def _handle_phase_completed(self, checklist: ChatChecklist, session_data: SessionData, history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Completed' phase.
        This function is called after the user answers the final real-life question
        from the 'resolution' phase.

        It acknowledges their answer and then asks if they want to start a new topic
        or end the chat.
        """
        # 'user_prompt' is the child's answer to the real-life question.
        user_final_answer = user_prompt
        user_decision = None

        # --- This is the new prompt you asked for ---
        wrapped_prompt = (
            f"--- CONTEXT ---\n"
            f"The child just finished the story and answered the final real-life question.\n"
            f"Their answer was: \"{user_final_answer}\".\n"
            f"The learning loop for this story is now complete.\n\n"

            f"--- YOUR TASK ---\n"
            f"Your job is to wrap up the conversation and ask what's next.\n\n"

            f"1.  **Acknowledge Their Answer:** Start with a positive, brief reply to their answer.\n"
            f"2.  **Praise Them:** Give them a final \"Awesome job!\" for finishing the story.\n"
            f"3.  **Ask \"What's Next?\":** Ask them if they want to learn a **new topic**, choose a **new story**, OR if they are **all done for today**.\n"

            f"--- EXAMPLE RESPONSE ---\n"
            f"\"That's a great answer, Derk! Awesome job learning all about that with me today.\n\n"
            f"So, what would you like to do next?\n"
            f"We can pick a new topic, or we can say goodbye for now!\n\n"
        )

        # --- IMPORTANT ---
        # This function creates a logical branch. The *next* user_prompt will be
        # either "goodbye" or a new topic. You will need to update this
        # function (or your main router) to handle that choice.

        # For now, we mark the 'completed_phase' as done.
        checklist.phases.mark_done("completed_phase")

        return await self._call_ai(session_data, history, wrapped_prompt)

    async def _handle_choice_phase(self, checklist: ChatChecklist, session_data: SessionData, history: list[types.Content], user_prompt: str) -> str:
        """
        TURN 2: Handles the user's decision ("new topic", "new story", or "goodbye").
        'user_prompt' is their decision.
        """

        # 1. Evaluate the user's decision
        user_decision = await self.evaluator.handle_completed_lesson_phase(user_prompt, session_data)

        # 2. Route based on the decision
        if user_decision == "NEW_TOPIC":

            # The 'choices_to_send' logic in process_message will now catch this
            # and send the topic list.
            return "You got it! What new topic would you like to learn about today?"

        elif user_decision == "NEW_STORY":
            checklist.phases.mark_done("pick_new_topic")
            print(checklist.phases.get_current_phase())
            return "What story do you want to learn with next?"

        elif user_decision == "END_CONVERSATION":
            # Say goodbye and end the chat
            return "You got it! Thanks for learning with me today. See you next time! 👋"

        else: # "UNCLEAR"
            # The AI didn't understand. Ask again.
            # We "un-mark" the previous phase to stay in this loop.
            checklist.phases.mark_undone("completed_phase")
            return "I'm sorry, I didn't quite understand. Did you want to start a new topic, pick another story, or are we all done for today?"

    # --- 4. The AI Caller and Config Builder ---

    async def _call_ai(self, session_data: SessionData, history: list[types.Content], wrapped_prompt: str) -> str:
        """
        A single, central asynchronous function to make calls to the Google GenAI API.
        It constructs the chat configuration (including persona and safety settings)
        and sends the conversation history along with a specially constructed
        'wrapped_prompt' (which contains instructions for the AI's current task).

        Args:
            checklist (ChatChecklist): The current state checklist.
            history (list[types.Content]): The chat history.
            wrapped_prompt (str): The prompt containing the AI's specific task for this turn.

        Returns:
            str: The AI's generated response text, or an error message if the API call fails.
        """

        # 1. Get the simple, base system instruction (persona)
        chat_config = self._get_chat_config(session_data)

        # 2. Retrieve the persistent chat session
        # This will now get the *existing* chat session, not create a new one every time.
        chat_session = session_data.onboarding_data.get("chat_session") # Assuming it's stored here now
        if not chat_session:
            # Fallback/Error: This should ideally not happen if _get_or_create_chat_session is called first.
            print("Error: Chat session not found in session data. Re-creating.")
            chat_session = await self._get_or_create_chat_session(session_data, history) # Re-create if missing


        # 3. Call the AI
        try:
            # Use the existing chat session to send the message.
            # The 'history' argument is no longer needed here as the session manages its own history.
            response = await chat_session.send_message(wrapped_prompt)
            return getattr(response, "text", "(SPARKY is thinking...)")

        except Exception as e:
            print(f"Error calling GenAI: {e}")
            return "Oh no! I got a little stuck. Can you try saying that again?"

    def _get_chat_config(self, session_data: SessionData) -> types.GenerateContentConfig:
        """
        Constructs the `GenerateContentConfig` for the GenAI API call.
        This includes the system instruction (SPARKY's persona and general guidelines)
        and safety settings.

        Returns:
            types.GenerateContentConfig: The configuration object for the GenAI model.
        """
        system_instruction = (
            f"You are a friendly Grade 3 peer tutor named SPARKY. You are talking to {session_data.onboarding_data.get('name')}.\n"
            f"You help them learn about {session_data.onboarding_data.get('chosen_topic')} through fun stories and experiments.\n\n"
            f"The main learning outcome for this topic is : {session_data.onboarding_data.get('story_learning_outcome')}.\n\n"
            "Guidelines:\n"
            "- Speak simply and kindly, like a curious classmate.\n"
            "- Use short sentences (8-12 words) and age-appropriate vocabulary.\n"
            "- Give hints or gentle feedback if the learner struggles. **NEVER** give the answer away if they say 'I don't know'. Always guide them.\n"
            "- Praise correct answers and relate ideas to real life when possible.\n"
            "- When narrating stories, if SPARKY is the character speaking, use first-person.\n"
        )

        # Define safety settings to control content generation.
        # These are set to "OFF" for specific categories, indicating a custom handling or
        # a less restrictive approach for this educational context.
        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ]

        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            safety_settings=safety_settings
        )

    async def _get_or_create_chat_session(self, session_data: SessionData, history: list[types.Content]) -> genai.chats.Chat:
        """
        Retrieves an existing chat session from the checklist or creates a new one
        if it doesn't exist. This ensures session persistence across turns.
        """
        chat_session = session_data.onboarding_data.get("chat_session")

        if chat_session is None:
            print("Creating new chat session...")
            chat_config = self._get_chat_config(session_data)
            chat_session = self.client.aio.chats.create(
                model=self.model_uri,
                config=chat_config,
                history=history
            )
            session_data.onboarding_data["chat_session"] = chat_session # Store it in the session data for persistence

        return chat_session

    # --- 5. Helper Methods ---
    def _find_story_by_id(self, story_id: str) -> dict | None:
        """
        Helper function to search through the master `self.story_data` list.
        """
        for story in self.stories_data:
            if story["story_id"] == story_id:
                return story
        return None

    def _get_story_phase_data(self, story_id: str, phase_name: str) -> dict | None:
        """
        Helper function to get phase data from a story.
        """
        story = self._find_story_by_id(story_id) # Uses the helper above
        if story and "phases" in story and phase_name in story["phases"]:
            return story["phases"][phase_name]
        return None

    def get_topic_list(self) -> list[str]:
        """
        Returns a list of available topics from the topics data.
        This version safely handles missing keys.
        """
        # 1. Use .get("topic_name") which returns None if the key is missing
        names = [topic.get("topic_name") for topic in self.topics_data]

        # 2. Filter out any 'None' values that might have been added
        return names

    def get_story_list_for_topic(self, topic_name: str) -> list[str]:
        """
        Retrieves a list of story names for a given topic.

        Args:
            topic_name: The name of the topic to search for.

        Returns:
            A list of story names, or an empty list if the topic
            or stories aren't found.
        """

        # Filter stories based on the chosen topic and prepare them for display.
        topic_stories = [story for story in self.story_data if story["topic"] == topic_name]
        story_map = {} # Maps display number to actual story_id
        story_display_list = []
        for i, story in enumerate(topic_stories, 1):
            story_map[i] = story["story_id"]
            story_display_list.append(f"{story['title']}")

        return story_display_list

    def get_topic_details(self, topic_name: str) -> dict | None:
        """
        Returns the details of a specific topic by name.
        Safely handles missing keys and returns None if not found.
        """
        for topic in self.topics_data:
            if topic.get("topic_name") == topic_name:
                return topic
        return None