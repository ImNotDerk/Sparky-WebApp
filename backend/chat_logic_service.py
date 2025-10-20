# chat_logic_service.py

from google import genai
from google.genai import types
import re # <-- Import regex

from checklist_manager import ChatChecklist
from input_evaluator import InputEvaluator

# Define a pattern to find our special token
PHASE_TOKEN_REGEX = re.compile(r"\[PHASE_ADVANCE:(\w+)\]")

class ChatLogicService:
    """
    Handles all the business logic for processing a chat message.
    It is stateless and receives all required data per call
    """
    def __init__(self, genai_client, model_uri: str, input_evaluator: InputEvaluator, story_data: list[dict]):
        self.client = genai_client
        self.model_uri = model_uri
        self.evaluator = input_evaluator
        self.story_data = story_data

    async def process_message(
            self, checklist: ChatChecklist, history: list[types.Content], user_prompt: str
        ) -> tuple[str, ChatChecklist]:
        """
        Main "Router" function.
        It checks the user's state (from checklist) and calls the correct handler.
        Returns the (bot_reply, updated_checklist)
        """

        next_step = checklist.next_step()
        bot_reply = None

        # --- MODIFIED LOGIC ---

        # 1.  Handle onboarding steps with static logic
        if next_step == "got_name":
            bot_reply = self._handle_get_name(checklist, user_prompt)
        elif next_step == "picked_topic":
            bot_reply = self._handle_pick_topic(checklist, user_prompt)
        elif next_step == "story_selected":
            bot_reply = self._handle_select_story(checklist, user_prompt)

        # 2. Handle 'story_started' (user says "ready")
        elif next_step == "story_started":
            # User is ready. Mark this step done.
            checklist.mark_done("story_started")
            # We will now call the AI to get the *first* story phase.
            # We pass an empty prompt because the AI's instruction (from config)
            # is to *start* the story, not reply to the user.
            bot_reply = await self._handle_main_continuation(checklist, history, "")
        
        # 3. Handle all other (post-onboarding) messages
        else:
            bot_reply = await self._handle_main_continuation(checklist, history, user_prompt)
        
        # --- END MODIFIED LOGIC ---

        # 4. Handle fallback
        if bot_reply is None:
            bot_reply  = "I'm not sure how to respond. Please try again or reset our chat."

        # 5. --- NEW: Check for and process phase-advance tokens ---
        match = PHASE_TOKEN_REGEX.search(bot_reply)
        if match:
            new_phase = match.group(1)
            checklist.data["current_phase"] = new_phase
            # Remove the token from the reply before sending to the user
            bot_reply = PHASE_TOKEN_REGEX.sub("", bot_reply).strip()

        # 6. Return the reply and the modified checlist
        return bot_reply, checklist

    # --- Private Handlers for each step --- #

    def _handle_get_name(self, checklist: ChatChecklist, prompt: str) -> str:
        """Handles the 'got_name' step."""
        if self.evaluator.is_empty_name_phrase(prompt):
            return "Oops! I didn't catch your name. Can you say it again? 😊"

        name = self.evaluator.extract_name(prompt)
        if name:
            checklist.data["child_name"] = name
            checklist.mark_done("got_name")
            return f"Nice to meet you, {name}! What would you like to learn about today?"
        else:
            return "Before we start, can you please tell me your name?"

    def _find_story_by_id(self, story_id: str) -> dict | None:
        """Helper to find a story object from its ID."""
        for story in self.story_data:
            if story["story_id"] == story_id:
                return story
        return None

    def _handle_pick_topic(self, checklist: ChatChecklist, prompt: str) -> str:
        """Handles the 'picked_topic' step."""
        if self.evaluator.is_empty_topic_phrase(prompt):
            return "Hmm, what topic would you like to learn about today?"

        topic = self.evaluator.extract_topic(prompt)
        if topic:
            checklist.data["topic"] = topic
            checklist.mark_done("picked_topic")

            topic_stories = [story for story in self.story_data if story["topic"] == topic]
            story_map = {}
            story_display_list = []
            for i, story in enumerate(topic_stories, 1):
                story_map[i] = story["story_id"]
                story_display_list.append(f"{i}. {story['title']}")

            checklist.data["story_map"] = story_map # Save map for next step

            story_list_str = "\n".join(story_display_list)
            return (
                f"Great choice! We're going to learn about **{topic}**.\n\n"
                f"Here are the stories you can choose from:\n"
                f"{story_list_str}\n\n"
                f"Please type the number of the story you'd like to start with!"
            )
        else:
            return "I didn't quite get that. What story number would you like to learn about today?"

    def _handle_select_story(self, checklist: ChatChecklist, prompt: str) -> str:
        """Handles the 'story_selected' step."""
        story_choice_num = self.evaluator.extract_story_choice(prompt)
        story_map = checklist.data.get("story_map")

        if story_map and story_choice_num in story_map:
            actual_story_id = story_map[story_choice_num]
            story = self._find_story_by_id(actual_story_id)
            
            checklist.data["story_choice"] = actual_story_id
            checklist.data["current_story_obj"] = story  # Store the whole story
            checklist.data["current_phase"] = "entry"  # Set the first phase
            checklist.mark_done("story_selected")

            story_title = story["title"] if story else "your chosen story"
            return f"Great choice! Let's start our adventure: \"{story_title}\". Are you ready?"
        else:
            return "I didn't quite get that. What topic would you like to learn about today?"

    # --- THIS IS NOW THE ONLY FUNCTION THAT CALLS THE AI ---
    async def _handle_main_continuation(self, checklist: ChatChecklist, history: list[types.Content], prompt: str) -> str:
        """
        Handles ALL AI-driven conversation, using the checklist to determine
        the correct system prompt (context) for the AI.
        """
        chat_config = self._get_chat_config(checklist)

        chat = self.client.aio.chats.create(
            model=self.model_uri,
            config=chat_config,
            history=history, # History *before* this user's prompt
        )
        
        # If prompt is empty, it means we are just starting the story
        if not prompt:
            prompt = "Start the story."

        response = await chat.send_message(prompt)
        return getattr(response, "text", "(no response generated)")

    def _get_chat_config(self, checklist: ChatChecklist) -> types.GenerateContentConfig:
        """
        A helper to build the model's configuration.
        This now dynamically builds the prompt based on the story phase.
        """
        
        # --- Base Prompt ---
        base_system_instruction = (
            f"You are a friendly Grade 3 peer tutor named SPARKY. You are talking to {checklist.data['child_name']}.\n"
            "Guidelines:\n"
            "- Speak simply and kindly, like a curious classmate.\n"
            "- Use short sentences (8-12 words) and age-appropriate vocabulary.\n"
            "- Give hints or gentle feedback if the learner struggles.\n"
            "- Praise correct answers and relate ideas to real life when possible.\n"
        )
        
        # --- Dynamic Task based on Phase ---
        current_phase = checklist.data.get("current_phase")
        story = checklist.data.get("current_story_obj")
        task_instruction = ""
        
        if story and current_phase and current_phase != "completed":
            phase_data = story["phases"].get(current_phase)
            if phase_data:
                story_text = phase_data["story"]
                question = phase_data["main_question"]
                
                # Determine the *next* phase to advance to
                next_phase_map = {
                    "entry": "engagement",
                    "engagement": "resolution",
                    "resolution": "completed"
                }
                next_phase = next_phase_map.get(current_phase, "completed")
                
                if current_phase == "entry":
                    # This is the first turn. The AI's job is to *start* the story.
                    task_instruction = (
                        f"\nTASK:\n"
                        f"Start the story '{story['title']}'.\n"
                        f"1. Tell this part of the story: \"{story_text}\"\n"
                        f"2. After telling the story part, ask this question: \"{question}\"\n"
                        f"3. At the very end of your message, add this exact token: [PHASE_ADVANCE:{next_phase}]"
                    )
                else:
                    # This is a subsequent turn. The AI's job is to *reply* and *then* continue.
                    task_instruction = (
                        f"\nTASK:\n"
                        f"The user just answered your last question. \n"
                        f"1. First, respond to their answer. (Praise them if correct, or gently guide them if they are wrong).\n"
                        f"2. After responding, tell them the next part of the story: \"{story_text}\"\n"
                        f"3. Then, ask the next question: \"{question}\"\n"
                        f"4. At the very end of your message, add this exact token: [PHASE_ADVANCE:{next_phase}]"
                    )
            else:
                # Fallback in case phase_data is missing
                current_phase = "completed"
        
        if current_phase == "completed":
            task_instruction = (
                f"\nTASK:\n"
                f"You have just finished the story '{story['title']}' about '{story['topic']}'.\n"
                f"Your new goal is to have a free-form conversation. Answer any follow-up questions the user has about the story or the topic. "
                "Help them understand the concepts. Do *not* add any phase tokens."
            )

        # Combine base + task
        system_instruction = base_system_instruction + task_instruction

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