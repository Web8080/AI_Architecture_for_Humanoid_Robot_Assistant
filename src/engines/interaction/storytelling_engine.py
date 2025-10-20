"""
Storytelling Engine - Interactive Story Generation

Creates and tells engaging stories for children and families.
Adapts stories based on age, interests, and preferences.
Supports interactive storytelling with user choices.

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
import time
import random
from typing import Dict, Any, List, Optional
from enum import Enum

from src.engines.base_engine import BaseEngine, EngineResponse, EngineStatus, EngineTier

logger = logging.getLogger(__name__)


class StoryType(Enum):
    """Types of stories"""
    ADVENTURE = "adventure"
    FAIRY_TALE = "fairy_tale"
    ANIMAL = "animal"
    SPACE = "space"
    MYSTERY = "mystery"
    EDUCATIONAL = "educational"
    BEDTIME = "bedtime"
    FUNNY = "funny"


class StorytellingEngine(BaseEngine):
    """
    Production-grade storytelling engine for home assistant
    
    Features:
    - Age-appropriate stories
    - Interactive storytelling (user choices)
    - Personalized characters (use child's name)
    - Multiple story types
    - Educational content integration
    - Moral lessons
    - Engaging narration
    
    Use Cases:
    - Bedtime stories
    - Entertainment
    - Educational stories
    - Interactive adventures
    - Moral lessons
    - Language learning
    
    Multi-tier fallback:
    - Tier 1: LLM-generated custom stories (unique every time)
    - Tier 2: Template-based stories with variations
    - Tier 3: Pre-written classic stories
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # LLM integration for Tier 1
        self.openai_client = None
        self.llama_client = None
        
        openai_key = config.get("openai_api_key") if config else None
        if openai_key and self.tier1_enabled:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_key)
                logger.info("OpenAI client initialized for storytelling")
            except Exception as e:
                logger.warning(f"OpenAI init failed: {e}")
        
        if self.tier2_enabled:
            try:
                import ollama
                self.llama_client = ollama
                logger.info("LLaMA client initialized for storytelling")
            except Exception as e:
                logger.warning(f"LLaMA init failed: {e}")
        
        # Story templates and classics
        self.story_templates = self._load_story_templates()
        self.classic_stories = self._load_classic_stories()
    
    def execute(self, entities: Dict[str, Any], context: Dict[str, Any]) -> EngineResponse:
        """
        Generate and tell a story
        
        Entities:
            - story_type: Type of story (optional)
            - character_name: Main character name (optional, defaults to child's name)
            - topic: Story topic/theme (optional)
            - length: short/medium/long (optional)
        
        Context:
            - user_age: Child's age
            - user_name: Child's name
            - user_interests: Known interests
            - previous_stories: Stories already told
        """
        start_time = time.time()
        
        if not self.enabled:
            return EngineResponse(
                status=EngineStatus.FAILURE,
                message="StorytellingEngine is disabled",
                data={},
                tier_used=EngineTier.TIER_3,
                execution_time=time.time() - start_time,
                confidence=0.0
            )
        
        return self._execute_with_fallback(
            self._tell_story_tier1,
            self._tell_story_tier2,
            self._tell_story_tier3,
            entities,
            context
        )
    
    def _tell_story_tier1(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Tier 1: LLM-generated custom story"""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not available")
        
        logger.info("Tier 1: Generating custom story with GPT-4")
        
        # Build story prompt
        story_prompt = self._build_story_prompt(entities, context)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert storyteller for children. Create engaging, age-appropriate, educational stories with moral lessons."},
                    {"role": "user", "content": story_prompt}
                ],
                max_tokens=1000,
                temperature=0.8
            )
            
            story_text = response.choices[0].message.content
            
            return {
                "message": story_text,
                "data": {
                    "story_type": entities.get("story_type", "custom"),
                    "generated": True,
                    "model": "gpt-4",
                    "tokens": response.usage.total_tokens
                },
                "confidence": 0.95
            }
        
        except Exception as e:
            logger.error(f"GPT-4 story generation failed: {e}")
            raise RuntimeError(f"Story generation failed: {e}")
    
    def _tell_story_tier2(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Tier 2: Template-based story with variations"""
        logger.info("Tier 2: Generating template-based story")
        
        story_type = entities.get("story_type", random.choice(list(StoryType)).value)
        character_name = entities.get("character_name") or context.get("user_name", "Alex")
        
        template = self.story_templates.get(story_type, self.story_templates["adventure"])
        
        # Fill in template with personalization
        story = template.format(
            name=character_name,
            age=context.get("user_age", 8),
            interest=context.get("user_interests", ["exploring"])[0] if context.get("user_interests") else "exploring"
        )
        
        return {
            "message": story,
            "data": {
                "story_type": story_type,
                "character_name": character_name,
                "method": "template"
            },
            "confidence": 0.85
        }
    
    def _tell_story_tier3(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Tier 3: Classic pre-written stories"""
        logger.info("Tier 3: Using classic story")
        
        story_type = entities.get("story_type", "bedtime")
        
        # Get classic story
        classic = random.choice(self.classic_stories.get(story_type, self.classic_stories["bedtime"]))
        
        return {
            "message": classic,
            "data": {
                "story_type": story_type,
                "method": "classic"
            },
            "confidence": 0.70
        }
    
    def _build_story_prompt(self, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build prompt for LLM story generation"""
        user_age = context.get("user_age", 8)
        user_name = context.get("user_name", "the child")
        story_type = entities.get("story_type", "adventure")
        topic = entities.get("topic", "")
        length = entities.get("length", "medium")
        
        prompt = f"""Create a {story_type} story for a {user_age}-year-old child.

Requirements:
- Main character name: {user_name}
- Story length: {length}
- Age-appropriate language and themes
- Include a moral lesson
- Make it engaging and fun
- Safe and positive content
"""
        
        if topic:
            prompt += f"- Story theme/topic: {topic}\n"
        
        if context.get("user_interests"):
            prompt += f"- Child's interests: {', '.join(context['user_interests'])}\n"
        
        prompt += "\nPlease create the story now."
        
        return prompt
    
    def _load_story_templates(self) -> Dict[str, str]:
        """Load story templates"""
        return {
            "adventure": """Once upon a time, there was a brave {age}-year-old named {name} who loved {interest}. 
One sunny morning, {name} discovered a mysterious map in the attic. The map showed a path to a hidden treasure!
{name} packed a backpack with snacks and water, and set off on an adventure. Along the way, {name} met friendly animals who helped solve riddles.
After many exciting challenges, {name} found the treasure - it wasn't gold or jewels, but a book filled with knowledge and wisdom!
{name} learned that the greatest treasures are the things we learn and the friends we make along the way.
The End.""",
            
            "space": """In a galaxy far, far away, lived a young space explorer named {name}. 
{name} had a special spaceship and loved learning about planets and stars.
One day, {name} received a message from a friendly alien who needed help! Their planet was running out of clean water.
{name} remembered learning about the water cycle and knew how to help! With science and teamwork, {name} helped the aliens build a system to collect and purify water.
The aliens were so grateful and invited {name} to visit their beautiful planet anytime. 
{name} learned that knowledge is powerful and helping others brings true happiness.
The End.""",
            
            "animal": """{name} loved animals more than anything! One day at the park, {name} noticed a little bird with a hurt wing.
{name} gently picked up the bird and brought it to an animal doctor. The doctor said the bird needed rest and care.
For weeks, {name} took care of the bird, feeding it and keeping it safe. Every day, {name} talked to the bird and the bird seemed to understand!
Finally, the bird's wing healed. When {name} released it, the bird flew in happy circles before landing on {name}'s shoulder to say goodbye.
{name} learned that kindness and patience can heal wounds, and that all creatures deserve our care and respect.
The End.""",
            
            "bedtime": """It was bedtime, and {name} was getting sleepy. As {name} snuggled under the cozy blankets, magical things began to happen!
The stars outside the window started twinkling in a special pattern, like they were dancing just for {name}.
The moon smiled down and whispered, 'Sweet dreams, {name}. Tomorrow will bring new adventures!'
{name} closed their eyes and dreamed of wonderful places - flying through clouds, playing with dolphins, and exploring castles.
In the dream, {name} met a wise owl who said, 'Rest well, young one. Sleep makes you strong for tomorrow's fun!'
{name} slept peacefully, knowing that every new day brings new joys and discoveries.
Good night! The End."""
        }
    
    def _load_classic_stories(self) -> Dict[str, List[str]]:
        """Load classic short stories"""
        return {
            "bedtime": [
                "Once there was a little star who was afraid of the dark. But then it learned that stars shine brightest in the darkness, bringing light and hope to everyone. Good night!",
                "A tiny seed was buried in the ground. It was scared and alone. But with patience, water, and sunshine, it grew into a beautiful flower that made everyone smile. Sleep tight!",
                "The moon and sun were best friends. Even though they couldn't be in the sky together, they knew they each had an important job - one brings light for play, one brings peace for sleep. Sweet dreams!"
            ],
            "adventure": [
                "A brave explorer climbed a tall mountain. It was hard work, but the view from the top was amazing! The explorer learned that challenges make victories sweeter.",
                "A curious explorer found a hidden cave full of crystals. Instead of taking them all, the explorer took just one to remember the adventure, leaving the rest for others to discover.",
            ],
            "animal": [
                "A lion had a thorn in its paw. A small mouse helped remove it. Later, when the lion was trapped, the mouse chewed through the ropes to save it. Even the smallest friend can be the biggest help!",
                "A tortoise and a hare raced. The hare was fast but took a nap. The slow tortoise kept going and won! Steady effort beats quick speed."
            ]
        }
    
    def get_capabilities(self) -> List[str]:
        return [
            "tell_story",
            "bedtime_story",
            "adventure_story",
            "educational_story",
            "interactive_story",
            "personalized_story",
            "age_appropriate_story",
            "llm_generated_story",
            "template_story",
            "classic_story"
        ]

