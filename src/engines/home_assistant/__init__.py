"""
Home Assistant Engines

Engines for household assistance, daily living support, and family interaction.
Inspired by Chapo bot but adapted for humanoid robot physical capabilities.

Author: Victor Ibhafidon
Date: November 2025
"""

from .weather_engine import WeatherEngine
from .news_engine import NewsEngine
from .alarm_engine import AlarmEngine
from .shopping_list_engine import ShoppingListEngine
from .music_engine import MusicEngine
from .calendar_engine import CalendarEngine
from .trivia_engine import TriviaEngine
from .smart_home_engine import SmartHomeEngine
from .pet_care_engine import PetCareEngine
from .cooking_assistant_engine import CookingAssistantEngine

__all__ = [
    'WeatherEngine',
    'NewsEngine',
    'AlarmEngine',
    'ShoppingListEngine',
    'MusicEngine',
    'CalendarEngine',
    'TriviaEngine',
    'SmartHomeEngine',
    'PetCareEngine',
    'CookingAssistantEngine',
]

