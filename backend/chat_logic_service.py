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
            self, checklist: ChatChecklist, history: list[types.Content], user_prompt: str
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

        next_step = checklist.next_step()
        bot_reply = None
        
        # === A. Handle Static Onboarding ===
        # These are initial, fixed steps to gather basic information from the user.
        if next_step == "got_name":
            bot_reply = self._handle_get_name(checklist, user_prompt)
        elif next_step == "picked_topic":
            bot_reply = self._handle_pick_topic(checklist, user_prompt)
        elif next_step == "story_selected":
            bot_reply = self._handle_select_story(checklist, user_prompt)

        # === B. Handle Dynamic Story Phases ===
        # Once onboarding is complete, the conversation enters dynamic story phases.
        else:
            current_phase = checklist.data.get("current_phase") 
            
            # Initialize the story if it's the very first dynamic turn.
            if not checklist.is_done("story_started"):
                checklist.mark_done("story_started")
                current_phase = "entry" # Force entry into the 'entry' phase
                user_prompt = "" # Clear user_prompt for the initial narration to prevent AI confusion

            # Route to the appropriate phase handler based on the current_phase.
            if current_phase == "entry":
                bot_reply = await self._handle_phase_entry(checklist, history, user_prompt)
            elif current_phase == "engagement":
                bot_reply = await self._handle_phase_engagement(checklist, history, user_prompt)
            elif current_phase == "experiment":
                bot_reply = await self._handle_phase_experiment(checklist, history, user_prompt)
            elif current_phase == "resolution":
                # The resolution phase includes logic for scaffolding, so it may not
                # advance immediately.
                bot_reply = await self._handle_phase_resolution(checklist, history, user_prompt)
            else: # 'completed' or an unknown phase
                bot_reply = await self._handle_phase_completed(checklist, history, user_prompt)

        # --- C. Final Processing ---
        # If no handler produced a reply, provide a fallback message.
        if bot_reply is None:
            bot_reply  = "I'm not sure how to respond. Please try again or reset our chat."

        # Check for and process the PHASE_ADVANCE token.
        # This token signals the system to transition to a new conversational phase.
        match = PHASE_TOKEN_REGEX.search(bot_reply)
        if match:
            new_phase = match.group(1)
            checklist.data["current_phase"] = new_phase # Update the current phase in the checklist
            bot_reply = PHASE_TOKEN_REGEX.sub("", bot_reply).strip() # Remove the token from the reply

        return bot_reply, checklist

    # --- 2. Static Onboarding Handlers ---

    def _handle_get_name(self, checklist: ChatChecklist, prompt: str) -> str:
        """
        Handles the 'got_name' onboarding step.
        Attempts to extract the child's name from the prompt and stores it.
        If successful, it moves to the next step; otherwise, it re-prompts for the name.
        """
        name = self.evaluator.extract_name(prompt)
        if name:
            checklist.data["child_name"] = name
            checklist.mark_done("got_name")
            return f"Nice to meet you, {name}! What would you like to learn about today?"
        return "Before we start, can you please tell me your name?"

    def _handle_pick_topic(self, checklist: ChatChecklist, prompt: str) -> str:
        """
        Handles the 'picked_topic' onboarding step.
        Extracts the topic from the user's prompt, stores it, and then presents
        a list of available stories for that topic.
        If the topic is not recognized, it re-prompts.
        """
        topic = self.evaluator.extract_topic(prompt)
        if topic:
            checklist.data["topic"] = topic
            checklist.mark_done("picked_topic")
            
            # Filter stories based on the chosen topic and prepare them for display.
            topic_stories = [story for story in self.story_data if story["topic"] == topic]
            story_map = {} # Maps display number to actual story_id
            story_display_list = []
            for i, story in enumerate(topic_stories, 1):
                story_map[i] = story["story_id"]
                story_display_list.append(f"{i}. {story['title']}")

            checklist.data["story_map"] = story_map # Store map for selection
            story_list_str = "\n".join(story_display_list)

            return (
                f"Great choice! We're going to learn about **{topic}**.\n\n"
                f"Here are the stories you can choose from:\n"
                f"{story_list_str}\n\n"
                f"Please type the number of the story you'd like to start with!"
            )
        return "I didn't quite get that. What topic would you like to learn about today?"

    def _handle_select_story(self, checklist: ChatChecklist, prompt: str) -> str:
        """
        Handles the 'story_selected' onboarding step.
        Extracts the story choice (number) from the user, retrieves the corresponding
        story data, and sets the initial dynamic phase to 'entry'.
        If the choice is invalid, it re-prompts.
        """
        story_choice_num = self.evaluator.extract_story_choice(prompt)
        story_map = checklist.data.get("story_map")

        if story_map and story_choice_num in story_map:
            actual_story_id = story_map[story_choice_num]
            story = self._find_story_by_id(actual_story_id)

            checklist.data["story_choice"] = actual_story_id
            checklist.data["current_story_obj"] = story  # Store the entire story object
            checklist.data["current_phase"] = "entry"    # Set the first dynamic phase
            checklist.mark_done("story_selected")

            story_title = story["title"] if story else "your chosen story"
            return f"Great choice! Let's start our adventure: \"{story_title}\". Are you ready?"
        return "I didn't quite get that. What story number would you like to learn about today?"

    # --- 3. Dynamic AI Phase Handlers ---
    # These functions manage the core learning experience, guiding the child
    # through observation, hypothesis, experiment, and conclusion.

    async def _handle_phase_entry(self, checklist: ChatChecklist, history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Entry Point' phase.
        SPARKY narrates the initial scene of the story and asks an open-ended
        question to encourage the child to make an observation.
        This phase transitions to 'engagement'.
        """
        story = checklist.data["current_story_obj"]
        phase_data = self._get_story_phase_data(story["story_id"], "entry")

        wrapped_prompt = (
            f"You are SPARKY. Start the story '{story['title']}'.\n"
            f"Your goal is to get the child to make an observation.\n\n"
            f"TASK:\n"
            f"1. Narrate this scene in your own simple, fun, first-person voice (as SPARKY):\n"
            f"   \"\"\"\n"
            f"   {phase_data['story']}\n"
            f"   \"\"\"\n"
            f"2. After narrating, ask an open-ended **observation question**.\n"
            f"   - **Good examples:** \"What do you notice about everyone in that story?\", \"What's something we are all doing?\", or \"What do you see happening?\"\n"
            f"   - **Bad example (don't use):** \"What do we all need?\"\n"
            f"3. At the very end of your message, add this exact token: [PHASE_ADVANCE:engagement]"
        )

        return await self._call_ai(checklist, history, wrapped_prompt)

    async def _handle_phase_engagement(self, checklist: ChatChecklist, history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Engagement' phase.
        SPARKY acknowledges the user's observation (from the previous 'entry' phase)
        and then asks a Socratic question to elicit their hypothesis related to the topic.
        This phase transitions to 'experiment'.
        """
        
        # The user_prompt in this phase is the child's observation from the 'entry' phase.
        user_observation = user_prompt 
        
        # Save the observation for potential future reference in other phases.
        checklist.data["last_observation"] = user_observation
        
        wrapped_prompt = (
            f"The user just made their observation: \"{user_observation}\"\n"
            f"TASK: Your job is to ask for their hypothesis (their 'guess').\n"
            f"1. Acknowledge their observation (e.g., \"That's a great observation, Derk! We were all feeling hungry.\").\n"
            f"2. Ask a simple, Socratic 'why' question to get their **hypothesis**. (e.g., \"When you feel hungry, what does your body need?\" or \"Why do you think we all need to do that?\").\n"
            f"3. **DO NOT** give them the answer or a hint yet. Focus purely on getting their initial guess.\n"
            f"4. **Add this token at the end:** [PHASE_ADVANCE:experiment]"
        )

        return await self._call_ai(checklist, history, wrapped_prompt)

    async def _handle_phase_experiment(self, checklist: ChatChecklist, history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Experiment' phase.
        This is a crucial phase where SPARKY dynamically generates a simple,
        two-part 'thought experiment' to test the child's hypothesis.
        The experiment generated is generalized and *must* be relevant to the current topic.
        It transitions to 'resolution'.
        """
        story = checklist.data["current_story_obj"]
        topic = checklist.data["topic"]
        
        # The user_prompt in this phase is the child's hypothesis from the 'engagement' phase.
        user_hypothesis = user_prompt
        
        # Save the hypothesis for comparison in the 'resolution' phase.
        checklist.data["last_hypothesis"] = user_hypothesis

        wrapped_prompt = (
            f"The user's hypothesis about our topic ('{topic}') is: \"{user_hypothesis}\"\n\n"
            f"--- TASK: Create a Generalized 'Thought Experiment' ---\n"
            f"Your goal is to test this hypothesis *specifically* for the topic of '{topic}'.\n"
            f"**DO NOT** change the topic. If the topic is 'Survival', the experiment must be about 'Survival'.\n\n"
            f"1.  **Acknowledge the Hypothesis:** Praise their guess (e.g., \"Wow, '{user_hypothesis}'... that's a smart hypothesis! Let's be scientists and test it! 🧑‍🔬\").\n"
            f"2.  **Propose the Experiment:** Create a simple, 2-part 'what if' scenario to test their idea. This experiment **must be directly related to '{topic}' and their hypothesis.**\n"
            f"    * **GENERAL PATTERN:** Imagine two similar 'things' (e.g., animals, plants, objects related to the topic). Give one 'thing' what it needs (based on the hypothesis/topic) and deprive the other 'thing' of it. The scenario should lead to a clear difference in outcome relevant to the topic.\n"
            f"    * **Example for 'Survival & Extinction':** \"Let's imagine two baby Dodos, Dodo 1 and Dodo 2. Dodo 1 has a warm, safe home high in a tree. Dodo 2's home was blown away by a storm. 🌪️\"\n"
            f"3.  **Ask for a Prediction:** Ask the child what they *think* will happen in your experiment (e.g., \"What do you *think* will happen to Dodo 1 and Dodo 2 tonight?\").\n"
            f"4.  **Add the Token:** [PHASE_ADVANCE:resolution]"
        )

        return await self._call_ai(checklist, history, wrapped_prompt)

    async def _handle_phase_resolution(self, checklist: ChatChecklist, history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Resolution' phase.
        This phase evaluates the child's prediction from the experiment.
        If the prediction is correct, SPARKY confirms, states the scientific conclusion
        related to the topic, tells the final story part, and asks a real-life question.
        If the prediction is incorrect or "I don't know", SPARKY provides scaffolding
        questions and re-prompts, ensuring the child arrives at the correct understanding.
        This phase only transitions to 'completed' once the child makes a reasonable prediction.
        """
        story = checklist.data["current_story_obj"]
        topic = checklist.data["topic"]
        phase_data = self._get_story_phase_data(story["story_id"], "resolution")
        
        # Retrieve the child's hypothesis from checklist data to compare against their prediction.
        last_hypothesis = checklist.data.get("last_hypothesis", "their guess")
        
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
        
        return await self._call_ai(checklist, history, wrapped_prompt)


    async def _handle_phase_completed(self, checklist: ChatChecklist, history: list[types.Content], user_prompt: str) -> str:
        """
        Handles the 'Completed' phase.
        Once the story and learning journey are complete, SPARKY engages in free-form chat.
        This function also cleans up any phase-specific flags (like `resolution_lesson_asked`)
        to ensure a clean slate for future stories.
        """
        story = checklist.data["current_story_obj"]
        
        wrapped_prompt = (
            f"The story '{story['title']}' is now complete. Have a fun, free-form chat with the user. "
            f"Respond to their last message: \"{user_prompt}\""
        )
        
        return await self._call_ai(checklist, history, wrapped_prompt)

    # --- 4. The AI Caller and Config Builder ---

    async def _call_ai(self, checklist: ChatChecklist, history: list[types.Content], wrapped_prompt: str) -> str:
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
        chat_config = self._get_chat_config(checklist)

        # 2. Call the AI
        try:
            # Create a new chat session for each turn to allow flexible configuration per turn.
            chat = self.client.aio.chats.create(
                model=self.model_uri,
                config=chat_config,
                history=history, # Pass the entire conversation history
            )
            # 3. Send the "wrapped_prompt" as the user's message.
            # The AI interprets its persona (from config) and its specific task (from wrapped_prompt).
            response = await chat.send_message(wrapped_prompt)
            return getattr(response, "text", "(SPARKY is thinking...)") # Extract text or provide placeholder

        except Exception as e:
            # Log the error for debugging purposes.
            print(f"Error calling GenAI: {e}")
            return "Oh no! I got a little stuck. Can you try saying that again?"

    def _get_chat_config(self, checklist: ChatChecklist) -> types.GenerateContentConfig:
        """
        Constructs the `GenerateContentConfig` for the GenAI API call.
        This includes the system instruction (SPARKY's persona and general guidelines)
        and safety settings.

        Args:
            checklist (ChatChecklist): The current state checklist, used to personalize
                                       the system instruction (e.g., child's name).

        Returns:
            types.GenerateContentConfig: The configuration object for the GenAI model.
        """
        system_instruction = (
            f"You are a friendly Grade 3 peer tutor named SPARKY. You are talking to {checklist.data.get('child_name', 'your friend')}.\n"
            "Guidelines:\n"
            "- Speak simply and kindly, like a curious classmate. Use emojis! 🥳\n"
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

    # --- 5. Helper Functions ---
    
    def _find_story_by_id(self, story_id: str) -> dict | None:
        """
        Helper function to search through `story_data` and find a story dictionary by its ID.

        Args:
            story_id (str): The unique identifier of the story.

        Returns:
            dict | None: The story dictionary if found, otherwise None.
        """
        for story in self.story_data:
            if story["story_id"] == story_id:
                return story
        return None

    def _get_story_phase_data(self, story_id: str, phase_name: str) -> dict | None:
        """
        Helper function to retrieve the specific data for a given phase within a story.

        Args:
            story_id (str): The ID of the story.
            phase_name (str): The name of the phase (e.g., 'entry', 'engagement', 'resolution').

        Returns:
            dict | None: The dictionary containing data for the specified phase, or None if not found.
        """
        story = self._find_story_by_id(story_id)
        if story and "phases" in story and phase_name in story["phases"]:
            return story["phases"][phase_name]
        return None