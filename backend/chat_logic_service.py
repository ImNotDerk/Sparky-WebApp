from google import genai
from google.genai import types

from checklist_manager import ChatChecklist
from input_evaluator import InputEvaluator

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

        # 1.  Route to the correct logic handler based on state
        if next_step == "got_name":
            bot_reply = self._handle_get_name(checklist, user_prompt)
        elif next_step == "picked_topic":
            bot_reply = self._handle_pick_topic(checklist, user_prompt)
        elif next_step == "story_selected":
            bot_reply = self._handle_select_story(checklist, user_prompt)
        elif next_step == "story_started":
            # This is an API call, so it's async
            bot_reply = await self._handle_start_story(checklist, history, user_prompt)
        else:
            # All onboarding steps are done, continue the main chat
            bot_reply = await self._handle_main_continuation(checklist, history, user_prompt)

        # 2. Handle fallback
        if bot_reply is None:
            bot_reply  = "I'm not sure how to respond. Please try again or reset our chat."

        # 3. Return the reply and the modified checlist
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

            # 1. Find all stories that match the chosen topic
            topic_stories = [story for story in self.story_data if story["topic"] == topic]

            # 2. Create a temporary mapping to show the user
            #    e.g., {1: "LT01-1", 2: "LT01-2", 3: "LT01-3"}
            #    And a list of titles to display
            story_map = {}
            story_display_list = []
            for i, story in enumerate(topic_stories, 1):
                story_map[i] = story["story_id"]
                story_display_list.append(f"{i}. {story['title']}")

            # 3. Save this map in the checklist for the *next* step
            checklist.data["story_map"] = story_map

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
        if self.evaluator.is_empty_topic_phrase(prompt):
            return "Hmm, that number isn't in the list. What topic would you like to learn about today?"

        # 1. Get the number the user typed (e.g., 1, 2, or 3)
        story_choice_num = self.evaluator.extract_story_choice(prompt)

        # 2. Get the story_map we saved in the previous step
        story_map = checklist.data.get("story_map")

        # 3. Check if the user's number is a valid key in our map
        if story_map and story_choice_num in story_map:

            # 4. Get the *actual* story_id (e.g., "LT01-1")
            actual_story_id = story_map[story_choice_num]

            # 5. Save this ID to the checklist
            checklist.data["story_choice"] = actual_story_id
            checklist.mark_done("story_selected")

            # 6. Find the story's title to confirm with the user
            story = self._find_story_by_id(actual_story_id)
            story_title = story["title"] if story else "your chosen story"

            return f"Great choice! Let's start our adventure: \"{story_title}\". Are you ready?"
        else:
            return "I didn't quite get that. What topic would you like to learn about today?"

    async def _handle_start_story(self, checklist: ChatChecklist, history: list[types.Content], prompt: str) -> str:
        """Handles the 'story_started' step."""
        checklist.mark_done("story_started")

        # This is the only place (besides continuation) wher we call the GenAI API
        chat_config = self._get_chat_config(checklist)

        chat = self.client.aio.chats.create(
            model=self.model_uri,
            config=chat_config,
            history=history, # History *before* this user's prompt
        )

        response = await chat.send_message(prompt)
        return getattr(response, "text", "(no response generated)")

    async def _handle_main_continuation(self, checklist: ChatChecklist, history: list[types.Content], prompt: str) -> str:
        """Handles the main chat continuation after onboarding is done."""
        chat_config = self._get_chat_config(checklist)

        chat = self.client.aio.chats.create(
            model=self.model_uri,
            config=chat_config,
            history=history, # History *before* this user's prompt
        )

        response = await chat.send_message(prompt)
        return getattr(response, "text", "(no response generated)")

    def _get_chat_config(self, checklist: ChatChecklist) -> types.GenerateContentConfig:
        """A helper to build the model's configuration."""
        story_id = checklist.data.get("story_choice")
        story = self._find_story_by_id(story_id) if story_id else None
        story_title = story["title"] if story else "a story"

        system_instruction = (
            f"You are a friendly Grade 3 peer tutor named SPARKY."
            f"You will be teaching {checklist.data['child_name']} Science concepts about \"{checklist.data['topic']}\" by telling them the story \"{story_title}\"."
            """
            Guidelines:
            - Speak simply and kindly, like a curious classmate.
            - Teach only within the given topic (e.g., "Living vs. Non-Living Things").
            - If the learner talks about something else, briefly acknowledge it then bring them back to the topic.
            - Use short sentences (8-12 words) and age-appropriate vocabulary.
            - Tell the story naturally, do not include section labels like (ENTRY POINT) or (ENGAGEMENT).
            - After each short story part, ask one friendly question that fits the lesson.
            - Give hints or gentle feedback if the learner struggles.
            - Praise correct answers and relate ideas to real life when possible.

            Goal:
            Help learners understand science concepts by guiding them through simple, story-based conversations that match their learning level.
            """
        )

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