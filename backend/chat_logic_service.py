# chat_logic_service.py

"""
This module defines the ChatLogicService, which orchestrates the conversational flow
of the Sparky chatbot. It manages the interaction phases (onboarding, story progression,
experimentation, and resolution) by routing user inputs to appropriate handlers and
generating AI responses. The service is designed to be stateless, relying on the
ChatChecklist to maintain session state.
"""

from google import genai
from google.genai import types
import re

from checklist_manager import ChatChecklist
from input_evaluator import InputEvaluator
from session_data_manager import SessionData

# Define a regular expression pattern to find our special phase advance token.
# This token is used internally to signal a transition to the next conversation phase.
PHASE_TOKEN_REGEX = re.compile(r"\[PHASE_ADVANCE:(\w+)\]")

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
    def __init__(self, genai_client, model_uri: str, input_evaluator: InputEvaluator, story_data: list[dict]):
        """
        Initializes the ChatLogicService with necessary dependencies.

        Args:
            genai_client: An instance of the Google GenAI client for API calls.
            model_uri (str): The URI of the generative AI model to use.
            input_evaluator (InputEvaluator): An instance to parse and evaluate user inputs.
            story_data (list[dict]): A list of dictionaries containing all available story data,
                                     including phases and their content.
        """
        self.client = genai_client
        self.model_uri = model_uri
        self.evaluator = input_evaluator
        self.story_data = story_data

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

        # === A. Handle Static Onboarding ===
        # These are initial, fixed steps to gather basic information from the user.
        if current_phase == "got_name":
            bot_reply = self._handle_get_name(checklist, session_data, user_prompt)
        elif current_phase == "picked_topic":
            bot_reply = self._handle_pick_topic(checklist, session_data, user_prompt)
        elif current_phase == "story_selected":
            bot_reply = self._handle_select_story(checklist, session_data, user_prompt)

            print("current phase:", checklist.phases.get_current_phase())

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
        else: # 'completed' or an unknown phase
            bot_reply = await self._handle_phase_completed(checklist, session_data, history, user_prompt)

        # --- C. Final Processing ---
        # If no handler produced a reply, provide a fallback message.
        if bot_reply is None:
            bot_reply  = "I'm not sure how to respond. Please try again or reset our chat."

        print("current phase:", checklist.phases.get_current_phase())

        return bot_reply, checklist, session_data

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
            return f"Nice to meet you, {name}! What would you like to learn about today? \n\n {self.evaluator.valid_topics}" # temporary
        return "Before we start, can you please tell me your name?"

    def _handle_pick_topic(self, checklist: ChatChecklist, session_data: SessionData, prompt: str) -> str:
        """
        Handles the 'picked_topic' onboarding step.
        Extracts the topic from the user's prompt, stores it, and then presents
        a list of available stories for that topic.
        If the topic is not recognized, it re-prompts.
        """
        topic = self.evaluator.extract_topic(prompt)
        if topic:
            session_data.onboarding_data["chosen_topic"] = topic

            # Filter stories based on the chosen topic and prepare them for display.
            topic_stories = [story for story in self.story_data if story["topic"] == topic]
            story_map = {} # Maps display number to actual story_id
            story_display_list = []
            for i, story in enumerate(topic_stories, 1):
                story_map[i] = story["story_id"]
                story_display_list.append(f"{i}. {story['title']}")

            session_data.onboarding_data["story_map"] = story_map # Store map for selection
            story_list_str = "\n".join(story_display_list)

            checklist.phases.mark_done("picked_topic")  # Mark this phase as done

            return (
                f"Great choice! We're going to learn about **{topic}**.\n\n"
                f"Here are the stories you can choose from:\n"
                f"{story_list_str}\n\n"
                f"Please type the number of the story you'd like to start with!"
            )
        return "I didn't quite get that. What topic would you like to learn about today?"

    def _handle_select_story(self, checklist: ChatChecklist, session_data: SessionData, prompt: str) -> str:
        """
        Handles the 'story_selected' onboarding step.
        Extracts the story choice (number) from the user, retrieves the corresponding
        story data, and sets the initial dynamic phase to 'entry'.
        If the choice is invalid, it re-prompts.
        """
        story_choice_num = self.evaluator.extract_story_choice(prompt)
        story_map = session_data.onboarding_data.get("story_map")

        if story_map and story_choice_num in story_map:
            actual_story_id = story_map[story_choice_num]
            story = self._find_story_by_id(actual_story_id)

            if story:

                session_data.onboarding_data["chosen_story"] = actual_story_id
                session_data.onboarding_data["story_data"] = story  # Store the entire story object
                checklist.phases.mark_done("story_selected")        # Mark this phase as done

                story_title = story["title"]
                return f"Great choice! Let's start our adventure: \"{story_title}\". Are you ready?"

        return "I didn't quite get that. What story number would you like to learn about today?"

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
        story = session_data.onboarding_data["story_data"] # Get Data for the chosen story
        phase_data = self._get_story_phase_data(story["story_id"], "entry") # Get Data for Entry Phase

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
            )

            checklist.sub_phases.mark_done("initial_entry_prompt")
            return await self._call_ai(session_data, history, wrapped_prompt)

        correct = self.evaluator.is_answer_correct(user_prompt, phase_data["expected_answer"])

        if correct: # if the child got the observation right, pass their observation to the next phase
            checklist.phases.mark_done("entry_point_phase") # Mark this phase as done
            checklist.sub_phases.mark_undone("initial_entry_prompt")
            return await self._handle_phase_engagement(checklist, session_data, history, user_prompt)

        else:
            wrapped_prompt = (
                f"The child tried to answer the observation question but wasn't quite right.\n"
                f"Their previous answer was: \"{user_prompt}\"\n"
                f"Entry point story reminder:\n\"{phase_data['story']}\"\n\n"
                f"TASK:\n"
                f"1. Encourage them to think again kindly.\n"
                f"2. Give a hint related to this scene.\n"
                f"3. Re-ask the question in a simpler way: {phase_data['main_question']}\n"
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
        phase_data = self._get_story_phase_data(story["story_id"], "engagement") # Get Data for Engagement Phase

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
                f"2. Ask a simple, Socratic 'why' question to get their **hypothesis**. (e.g., \"{phase_data['main_question']}\").\n"
            )

            checklist.sub_phases.mark_done("initial_engagement_prompt") # Set initial engagement prompt to done
            return await self._call_ai(session_data, history, wrapped_prompt)

        # If the flag is True, this is a SUBSEQUENT call.
        # The 'user_prompt' is now the child's HYPOTHESIS.
        else:
            user_hypothesis = user_prompt

            correct = self.evaluator.is_answer_correct(user_hypothesis, phase_data["expected_answer"])

            if correct:
                # The hypothesis is correct!
                # 1. Reset the flag for the next time we run this story
                checklist.phases.mark_done("engagement_phase") # Mark this phase as done
                checklist.sub_phases.mark_undone("initial_engagement_prompt") # Mark initial engagement prompt as undone
                print("Hypothesis correct, moving to experiment phase.")
                # 2. Manually call the NEXT phase handler
                return await self._handle_phase_experiment(checklist, session_data, history, user_hypothesis)

            else:
                # The hypothesis is incorrect, "I don't know," or silly.
                # We scaffold and stay in this phase.
                print("Hypothesis incorrect, scaffolding.")
                wrapped_prompt = (
                    f"The child tried to answer the hypothesis question but wasn't quite right.\n"
                    f"Their previous guess was: \"{user_prompt}\"\n"
                    f"The current topic is: \"{session_data.onboarding_data['chosen_topic']}\"\n"
                    f"Story reminder:\n\"{phase_data['story']}\"\n\n"
                    f"TASK:\n"
                    f"1. Encourage them to think again kindly (e.g., \"That's an interesting guess! Let's think about it...\").\n"
                    f"2. Give a hint related to the topic and the story scene.\n"
                    f"3. Re-ask the hypothesis question in a simpler way: {phase_data['main_question']}\n"
                )

                return await self._call_ai(session_data, history, wrapped_prompt)

    async def _handle_phase_experiment(self, checklist: ChatChecklist, session_data: SessionData, history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Experiment' phase in two steps:
        1. Evaluates the child's hypothesis. If valid, asks them to propose an experiment.
        2. Evaluates the child's proposed experiment. If invalid, SPARKY proposes its own.
        3. Asks for a prediction for the chosen experiment.
        """
        story = session_data.onboarding_data["story_data"]
        topic = session_data.onboarding_data["chosen_topic"]

        # Get the sub-state flag
        asked_for_child_experiment = checklist.sub_phases.is_done("initial_experiment_prompt")

        if not asked_for_child_experiment:
            # --- This is the FIRST call to this phase ---
            # 'user_prompt' is the child's (presumed valid) HYPOTHESIS
            user_hypothesis = user_prompt
            session_data.important_conversation_data["last_hypothesis"] = user_hypothesis

            wrapped_prompt = (
                f"The user's hypothesis about our topic ('{topic}') is: \"{user_hypothesis}\"\n\n"
                f"--- TASK: Ask for Child's Experiment ---\n"
                f"1.  **Acknowledge the Hypothesis:** Praise their guess (e.g., \"Wow, '{user_hypothesis}'... that's a smart hypothesis! 🧑‍🔬\").\n"
                f"2.  **Ask for Experiment:** Now, ask the child if *they* can think of a simple 'what if' experiment to test their idea. (e.g., \"How could we test that? Can you think of a 'what if' experiment for us to try?\")\n"
            )

            checklist.sub_phases.mark_done("initial_experiment_prompt")
            return await self._call_ai(session_data, history, wrapped_prompt)

        # fix the logic here
        else:
            user_experiment_idea = user_prompt

            is_valid_experiment = await self.evaluator.is_experiment_valid(user_experiment_idea, session_data)

            if is_valid_experiment: # ask for a conclusion or what they learned
                # The child's experiment idea is valid!
                # Proceed to ask for their prediction.
                wrapped_prompt = (
                    f"The child proposed a valid experiment idea: \"{user_experiment_idea}\"\n\n"
                    f"--- TASK: Ask for Prediction ---\n"
                    f"1.  **Praise the Experiment Idea:** (e.g., \"That's a fantastic experiment! Let's see what happens!\").\n"
                    f"2.  **Restate the Experiment Clearly:** Make sure to clearly restate their experiment idea.\n"
                    f"3.  **Ask for Prediction:** Now, ask them: \"What do you *think* will happen in your experiment?\"\n"
                )

                # Reset the flag so this phase works correctly next
                # checklist.phases.mark_done("experimental_phase") # Mark this phase as done
                # checklist.sub_phases.mark_undone("initial_experiment_prompt")

                return await self._call_ai(session_data, history, wrapped_prompt)
            # --- This is the SECOND call to this phase ---
            # 'user_prompt' is the child's EXPERIMENT IDEA (or "i dont know")
            else: # Create our own experiment and ask for prediction
                user_hypothesis = session_data.important_conversation_data.get("last_hypothesis", "their guess") # Get the saved hypothesis

                # 'user_experiment_idea' and 'topic' are assumed to be defined
                # earlier in the function (e.g., from the 'if' check or session_data).

                wrapped_prompt = (
                    f"--- CONTEXT ---\n"
                    f"The user's hypothesis (their guess) is: \"{user_hypothesis}\".\n"
                    f"We asked them to create an experiment, but their response was: \"{user_experiment_idea}\".\n"
                    f"This means they are stuck or said 'I don't know'.\n"
                    f"The current learning topic is: \"{topic}\".\n\n"

                    f"--- YOUR TASK ---\n"
                    f"You must now take the lead. **Do not** evaluate their idea, just take over.\n\n"

                    f"1.  **Acknowledge and Reassure:** Start with a gentle, encouraging phrase.\n\n"

                    f"2.  **Propose YOUR OWN Experiment:** Create a simple, clear, 'what if' experiment. This experiment **must** be designed to test their hypothesis: \"{user_hypothesis}\".\n\n"

                    f"3.  **Relate it:** Make sure your experiment is clearly related to the topic: \"{topic}\". (If possible, try to tie it to the current story scene.)\n\n"

                    f"4.  **Make sure it's Testable:** The experiment should involve a clear action or comparison.\n\n"

                    f"5.  **Make sure it's scientifically accurate**: The information in the experiment should align with real scientific principles related to the topic.\n\n"

                    f"6.  **Ask for Prediction:** End by asking for their prediction about *your* experiment. (e.g., \"What do you *think* will happen in this experiment?\", \"What will be the result?\").\n\n"

                    f"7.  **Add Control Token:** You MUST add this exact token to the *very end* of your response (on its own line):\n"
                )

                # Reset the flag so this phase works correctly next
                checklist.phases.mark_done("experimental_phase") # Mark this phase as done
                checklist.sub_phases.mark_undone("initial_experiment_prompt")

                return await self._call_ai(session_data, history, wrapped_prompt)
            
    async def _handle_conclusion_phase(self, checklist: ChatChecklist, session_data: SessionData, history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Conclusion' phase.
        This phase wraps up the experiment by asking the child what they learned
        from the experiment and story. It encourages reflection and articulation
        of the scientific concept.
        """
        story = session_data.onboarding_data["story_data"]
        topic = session_data.onboarding_data["chosen_topic"]
        phase_data = self._get_story_phase_data(story["story_id"], "conclusion")

        wrapped_prompt = (
            f"The experiment has concluded. Now, it's time to reflect on what we've learned.\n\n"
            f"--- TASK: Ask for Conclusion ---\n"
            f"1.  **Prompt Reflection:** Ask the child what they learned from the experiment and story related to the topic '{topic}'.\n"
            f"2.  **Encourage Articulation:** Encourage them to explain in their own words (e.g., \"What did we find out? Can you tell me what you learned?\").\n"
        )

        checklist.phases.mark_done("conclusion_phase") # Mark this phase as done

        return await self._call_ai(session_data, history, wrapped_prompt)

    async def _handle_phase_resolution(self, checklist: ChatChecklist, session_data: SessionData, history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Resolution' phase.
        This phase evaluates the child's prediction from the experiment.
        If the prediction is correct, SPARKY confirms, states the scientific conclusion
        related to the topic, tells the final story part, and asks a real-life question.
        If the prediction is incorrect or "I don't know", SPARKY provides scaffolding
        questions and re-prompts, ensuring the child arrives at the correct understanding.
        This phase only transitions to 'completed' once the child makes a reasonable prediction.
        """
        story = session_data.onboarding_data["story_data"]
        topic = session_data.onboarding_data["chosen_topic"]
        phase_data = self._get_story_phase_data(story["story_id"], "resolution")

        # Retrieve the child's hypothesis from checklist data to compare against their prediction.
        last_hypothesis = session_data.important_conversation_data.get("last_hypothesis", "their guess")

        # The user_prompt in this phase is the child's prediction for the experiment.
        user_prediction = user_prompt

        wrapped_prompt = (
            f"The user's original hypothesis was: \"{last_hypothesis}\".\n"
            f"Your experiment just tested that. The user's prediction for your experiment was: \"{user_prediction}\"\n"
            f"The current topic is: \"{topic}\".\n\n"
            f"--- TASK: Evaluate the Prediction and Conclude ---\n"
            f"Your goal is to make the child *articulate* the lesson from the experiment and story.\n"
            f"**DO NOT** just tell them the answer or move on, especially if they say 'I don't know' (like in the problematic image_a093ab.png example).\n\n"

            f"1.  **Analyze the User's Prediction:** Determine if \"{user_prediction}\" is a reasonable outcome for your experiment, or if they expressed uncertainty ('I don't know').\n\n"

            f"2.  **IF THE PREDICTION IS REASONABLE (i.e., correct or very close to the scientific outcome):**\n"
            f"    a. Praise them! (e.g., \"You're exactly right! That's what would happen!\").\n"
            f"    b. **State the Conclusion:** Clearly articulate the scientific takeaway *that explicitly connects the original hypothesis, the experiment's outcome, and the topic '{topic}'*. (e.g., \"So that proves our hypothesis! Living things need a safe home to *survive*!\").\n"
            f"    c. **Tell the Resolution Story:** Now, smoothly transition and tell the final story part: \"{phase_data['story']}\"\n"
            f"    d. **Ask the Final Question:** Ask the real-life application question: \"{phase_data['main_question']}\"\n"
            f"    e. **Add this token at the end:** [PHASE_ADVANCE:completed] (This signals moving to the free-form 'completed' phase).\n\n"

            f"3.  **IF THE PREDICTION IS 'I DON'T KNOW' or UNREASONABLE (i.e., incorrect or not relevant to the experiment):**\n"
            f"    a. Gently acknowledge their answer (e.g., \"That's a great question!\" or \"It's okay not to know! Let's think...\").\n"
            f"    b. **Provide a Scaffold/Hint:** Give a hint *directly related to the specific experiment you just proposed*. Guide them by reminding them of the setup or asking a simpler, related question. (e.g., \"If Dodo 2's home blew away, where will it sleep tonight? What if it gets cold? What does that mean for its survival?\").\n"
            f"    c. **Re-ask the experiment's prediction question.** (e.g., \"So, what do you think will happen to Dodo 1 and Dodo 2 tonight?\").\n"
            f"    d. **DO NOT** add any phase token. This ensures the child remains in the 'resolution' phase to attempt the prediction again with guidance."
        )

        checklist.phases.mark_done("resolution_phase") # Mark this phase as done
        #checklist.sub_phases.mark_undone("initial_experiment_prompt")

        return await self._call_ai(session_data, history, wrapped_prompt)


    async def _handle_phase_completed(self, checklist: ChatChecklist, session_data: SessionData, history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Completed' phase.
        Once the story and learning journey are complete, SPARKY engages in free-form chat.
        This function also cleans up any phase-specific flags (like `resolution_lesson_asked`)
        to ensure a clean slate for future stories.
        """
        story = session_data.onboarding_data["story_data"]

        wrapped_prompt = (
            f"The story '{story['title']}' is now complete. Have a fun, free-form chat with the user. "
            f"Respond to their last message: \"{user_prompt}\""
        )

        return await self._call_ai(session_data, history, wrapped_prompt)

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
                history=history # Use the full history for the *initial* creation
            )
            session_data.onboarding_data["chat_session"] = chat_session # Store it in the session data for persistence

        return chat_session

    # --- 5. Helper Methods ---
    def _find_story_by_id(self, story_id: str) -> dict | None:
        """
        Helper function to search through the master `self.story_data` list.
        """
        for story in self.story_data:
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