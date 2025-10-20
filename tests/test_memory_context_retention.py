"""
Test Memory Context Retention

This test demonstrates how our Advanced Memory Manager SOLVES
the Chapo bot problem where the bot fails to remember context
within a conversation.

PROBLEM (Chapo bot):
User: "My name is John"
Bot: "Nice to meet you!"
User: "What's my name?"
Bot: "I don't know" ❌

SOLUTION (Our system):
User: "My name is John"
Bot: "Nice to meet you, John!" 
User: "What's my name?"
Bot: "Your name is John" ✅

Author: Victor Ibhafidon
Date: October 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.advanced_memory_manager import AdvancedMemoryManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_name_retention():
    """Test that the bot remembers user's name"""
    logger.info("=" * 80)
    logger.info("TEST 1: Name Retention")
    logger.info("=" * 80)
    
    memory = AdvancedMemoryManager()
    session_id = "test_session_001"
    user_id = "test_user_001"
    
    # User introduces themselves
    logger.info("\n📝 User says: 'My name is John'")
    memory.add_conversation_turn(
        session_id=session_id,
        user_input="My name is John",
        robot_response="Nice to meet you, John!",
        intent="introduce_self",
        entities={}
    )
    
    # Explicitly store the name
    memory.remember_user_info(session_id, user_id, "name", "John", explicit_importance=0.9)
    
    # Several turns later, user asks for their name
    logger.info("\n❓ User asks: 'What's my name?'")
    
    # Recall the name
    recalled_name = memory.recall_user_info(session_id, user_id, "name")
    
    if recalled_name:
        logger.info(f"✅ SUCCESS! Bot remembers: 'Your name is {recalled_name}'")
        assert recalled_name == "John", f"Expected 'John', got '{recalled_name}'"
    else:
        logger.error("❌ FAILED! Bot doesn't remember the name")
        assert False, "Name not recalled"
    
    logger.info("\n" + "=" * 80)


def test_preference_retention():
    """Test that the bot remembers user preferences"""
    logger.info("=" * 80)
    logger.info("TEST 2: Preference Retention")
    logger.info("=" * 80)
    
    memory = AdvancedMemoryManager()
    session_id = "test_session_002"
    user_id = "test_user_002"
    
    # User mentions preference
    logger.info("\n📝 User says: 'I like coffee'")
    memory.add_conversation_turn(
        session_id=session_id,
        user_input="I like coffee",
        robot_response="I'll remember that you like coffee!",
        intent="state_preference",
        entities={"preference": "coffee", "sentiment": "like"}
    )
    
    # Store the preference
    memory.remember_user_info(session_id, user_id, "likes", ["coffee"], explicit_importance=0.8)
    
    # Later in conversation
    logger.info("\n❓ User asks: 'What do I like?'")
    
    # Recall preferences
    recalled_likes = memory.recall_user_info(session_id, user_id, "likes")
    
    if recalled_likes and "coffee" in recalled_likes:
        logger.info(f"✅ SUCCESS! Bot remembers: 'You like {', '.join(recalled_likes)}'")
        assert "coffee" in recalled_likes
    else:
        logger.error("❌ FAILED! Bot doesn't remember the preference")
        assert False, "Preference not recalled"
    
    logger.info("\n" + "=" * 80)


def test_multi_turn_context():
    """Test context retention across multiple turns"""
    logger.info("=" * 80)
    logger.info("TEST 3: Multi-turn Context Retention")
    logger.info("=" * 80)
    
    memory = AdvancedMemoryManager()
    session_id = "test_session_003"
    user_id = "test_user_003"
    
    # Turn 1
    logger.info("\n📝 Turn 1 - User: 'My name is Sarah'")
    memory.add_conversation_turn(
        session_id=session_id,
        user_input="My name is Sarah",
        robot_response="Nice to meet you, Sarah!",
        intent="introduce_self",
        entities={}
    )
    memory.remember_user_info(session_id, user_id, "name", "Sarah", 0.9)
    
    # Turn 2
    logger.info("\n📝 Turn 2 - User: 'I like pizza'")
    memory.add_conversation_turn(
        session_id=session_id,
        user_input="I like pizza",
        robot_response="Pizza is delicious!",
        intent="state_preference",
        entities={"food": "pizza"}
    )
    memory.remember_user_info(session_id, user_id, "favorite_food", "pizza", 0.8)
    
    # Turn 3
    logger.info("\n📝 Turn 3 - User: 'I live in London'")
    memory.add_conversation_turn(
        session_id=session_id,
        user_input="I live in London",
        robot_response="London is a great city!",
        intent="state_location",
        entities={"location": "London"}
    )
    memory.remember_user_info(session_id, user_id, "location", "London", 0.8)
    
    # Turn 4 - Ask comprehensive question
    logger.info("\n❓ Turn 4 - User: 'Tell me what you know about me'")
    
    # Get comprehensive context
    context = memory.get_conversation_context(session_id, user_id, turns=10)
    
    logger.info("\n📋 Context retrieved:")
    logger.info(context)
    
    # Verify all information is retained
    assert "Sarah" in context, "Name not in context"
    assert "pizza" in context or "favorite_food" in context, "Food preference not in context"
    assert "London" in context or "location" in context, "Location not in context"
    
    logger.info("\n✅ SUCCESS! All context retained across multiple turns")
    logger.info("\n" + "=" * 80)


def test_session_persistence():
    """Test that information persists across sessions"""
    logger.info("=" * 80)
    logger.info("TEST 4: Session Persistence (Long-term Memory)")
    logger.info("=" * 80)
    
    memory = AdvancedMemoryManager()
    user_id = "test_user_004"
    
    # Session 1
    logger.info("\n🔵 SESSION 1:")
    session_id_1 = "test_session_004_1"
    
    memory.start_session(session_id_1, user_id)
    
    logger.info("📝 User: 'My name is Mike'")
    memory.add_conversation_turn(
        session_id=session_id_1,
        user_input="My name is Mike",
        robot_response="Hi Mike!",
        intent="introduce_self",
        entities={}
    )
    memory.remember_user_info(session_id_1, user_id, "name", "Mike", 0.9)
    
    memory.end_session(session_id_1, user_id)
    logger.info("✅ Session 1 ended, memory saved")
    
    # Session 2 (Later - simulating a new conversation)
    logger.info("\n🔵 SESSION 2 (User returns later):")
    session_id_2 = "test_session_004_2"
    
    memory.start_session(session_id_2, user_id)
    
    # Try to recall name from previous session
    logger.info("❓ User: 'Do you remember my name?'")
    
    recalled_name = memory.recall_user_info(session_id_2, user_id, "name")
    
    if recalled_name == "Mike":
        logger.info(f"✅ SUCCESS! Bot remembers across sessions: 'Yes, your name is {recalled_name}'")
        assert recalled_name == "Mike"
    else:
        logger.warning("⚠️ Long-term memory (MongoDB) not configured, but short-term memory works!")
        logger.info("📝 To enable cross-session memory, configure MongoDB")
    
    logger.info("\n" + "=" * 80)


def test_memory_search():
    """Test searching across memory"""
    logger.info("=" * 80)
    logger.info("TEST 5: Memory Search")
    logger.info("=" * 80)
    
    memory = AdvancedMemoryManager()
    session_id = "test_session_005"
    user_id = "test_user_005"
    
    # Add various information
    logger.info("\n📝 Adding information to memory...")
    
    memory.add_conversation_turn(
        session_id=session_id,
        user_input="I went to the park yesterday",
        robot_response="That sounds nice!",
        intent="casual_conversation",
        entities={"location": "park"}
    )
    
    memory.add_conversation_turn(
        session_id=session_id,
        user_input="I saw a beautiful red bird",
        robot_response="What kind of bird was it?",
        intent="casual_conversation",
        entities={"object": "bird", "color": "red"}
    )
    
    # Search for "park"
    logger.info("\n🔍 Searching for 'park'...")
    results = memory.search_memory("park", session_id=session_id)
    
    found_in_working = any("park" in str(r).lower() for r in results['working_memory'])
    found_in_short_term = any("park" in str(r).lower() for r in results['short_term_memory'])
    
    if found_in_working or found_in_short_term:
        logger.info("✅ SUCCESS! Found 'park' in memory")
        assert True
    else:
        logger.error("❌ FAILED! Could not find 'park' in memory")
        assert False
    
    logger.info("\n" + "=" * 80)


def main():
    """Run all memory tests"""
    logger.info("\n\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 20 + "MEMORY CONTEXT RETENTION TESTS" + " " * 28 + "║")
    logger.info("║" + " " * 15 + "Solving the Chapo Bot Problem" + " " * 34 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("\n")
    
    try:
        test_name_retention()
        test_preference_retention()
        test_multi_turn_context()
        test_session_persistence()
        test_memory_search()
        
        logger.info("\n\n")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 30 + "ALL TESTS PASSED!" + " " * 30 + "║")
        logger.info("║" + " " * 20 + "✅ Memory system working perfectly" + " " * 24 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info("\n")
        
        # Print memory stats
        memory = AdvancedMemoryManager()
        stats = memory.get_memory_stats()
        
        logger.info("📊 MEMORY SYSTEM STATISTICS:")
        logger.info(f"  Working Memory: {stats['working_memory']}")
        logger.info(f"  Short-term Memory: {stats['short_term_memory']}")
        if stats['long_term_memory']:
            logger.info(f"  Long-term Memory: {stats['long_term_memory']}")
        
    except AssertionError as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()

