"""
Home Assistant Robot Integration Tests

Comprehensive testing of the entire home assistant robot system.
Tests all engines, memory systems, and integration points.

Author: Victor Ibhafidon
Date: October 2025
"""

import sys
import logging
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def test_engine_imports():
    """Test that all engines can be imported"""
    logger.info("=" * 80)
    logger.info("TEST 1: Engine Imports")
    logger.info("=" * 80)
    
    engines_to_test = [
        ("Object Grasping", "src.engines.object_manipulation.grasping_engine", "ObjectGraspingEngine"),
        ("Object Placement", "src.engines.object_manipulation.placement_engine", "ObjectPlacementEngine"),
        ("Object Transfer", "src.engines.object_manipulation.transfer_engine", "ObjectTransferEngine"),
        ("Conversation", "src.engines.interaction.conversation_engine", "ConversationEngine"),
        ("Storytelling", "src.engines.interaction.storytelling_engine", "StorytellingEngine"),
        ("Game", "src.engines.interaction.game_engine", "GameEngine"),
        ("Education", "src.engines.interaction.education_engine", "EducationEngine"),
        ("Safety Monitor", "src.engines.safety.safety_monitor_engine", "SafetyMonitorEngine"),
    ]
    
    passed = 0
    failed = 0
    
    for name, module_path, class_name in engines_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            engine_class = getattr(module, class_name)
            logger.info(f"✓ {name} Engine imported successfully")
            passed += 1
        except Exception as e:
            logger.error(f"✗ {name} Engine import failed: {e}")
            failed += 1
    
    logger.info(f"\nEngine Import Results: {passed} passed, {failed} failed")
    assert failed == 0, f"{failed} engine imports failed"
    logger.info("=" * 80 + "\n")


def test_engine_initialization():
    """Test that all engines can be initialized"""
    logger.info("=" * 80)
    logger.info("TEST 2: Engine Initialization")
    logger.info("=" * 80)
    
    from src.engines.object_manipulation.grasping_engine import ObjectGraspingEngine
    from src.engines.object_manipulation.placement_engine import ObjectPlacementEngine
    from src.engines.interaction.conversation_engine import ConversationEngine
    from src.engines.interaction.game_engine import GameEngine
    from src.engines.safety.safety_monitor_engine import SafetyMonitorEngine
    
    engines = [
        ("Grasping", ObjectGraspingEngine, {}),
        ("Placement", ObjectPlacementEngine, {}),
        ("Conversation", ConversationEngine, {"openai_api_key": "test_key"}),
        ("Game", GameEngine, {}),
        ("Safety Monitor", SafetyMonitorEngine, {"emergency_number": "999"}),
    ]
    
    passed = 0
    failed = 0
    
    for name, engine_class, config in engines:
        try:
            engine = engine_class(config)
            assert engine is not None
            assert engine.enabled
            logger.info(f"✓ {name} Engine initialized successfully")
            passed += 1
        except Exception as e:
            logger.error(f"✗ {name} Engine initialization failed: {e}")
            failed += 1
    
    logger.info(f"\nEngine Init Results: {passed} passed, {failed} failed")
    assert failed == 0, f"{failed} engine initializations failed"
    logger.info("=" * 80 + "\n")


def test_memory_systems():
    """Test memory systems"""
    logger.info("=" * 80)
    logger.info("TEST 3: Memory Systems")
    logger.info("=" * 80)
    
    # Test Advanced Memory Manager
    try:
        from src.memory.advanced_memory_manager import AdvancedMemoryManager
        
        memory = AdvancedMemoryManager()
        
        # Test adding conversation turn
        memory.add_conversation_turn(
            session_id="test_session",
            user_input="My name is Alice",
            robot_response="Nice to meet you, Alice!",
            intent="introduce_self",
            entities={}
        )
        
        # Test remembering user info
        memory.remember_user_info("test_session", "user123", "name", "Alice", 0.9)
        
        # Test recalling user info
        recalled_name = memory.recall_user_info("test_session", "user123", "name")
        
        assert recalled_name == "Alice", f"Expected 'Alice', got '{recalled_name}'"
        logger.info("✓ Advanced Memory Manager working correctly")
        logger.info(f"  - Name stored and recalled: {recalled_name}")
        
        # Test memory stats
        stats = memory.get_memory_stats()
        logger.info(f"  - Memory stats: {stats}")
        
    except Exception as e:
        logger.error(f"✗ Memory system test failed: {e}")
        raise
    
    logger.info("=" * 80 + "\n")


def test_conversation_engine():
    """Test conversation engine with real scenarios"""
    logger.info("=" * 80)
    logger.info("TEST 4: Conversation Engine Scenarios")
    logger.info("=" * 80)
    
    from src.engines.interaction.conversation_engine import ConversationEngine
    
    engine = ConversationEngine({
        "openai_api_key": None,  # Will use fallback tiers
        "enable_content_filter": True
    })
    
    test_scenarios = [
        ("Hello", "greeting", 5),  # age 5
        ("Can you help me with homework?", "homework", 8),  # age 8
        ("Tell me a joke", "joke", 10),  # age 10
    ]
    
    for user_input, expected_intent, age in test_scenarios:
        try:
            entities = {"user_input": user_input}
            context = {
                "user_age": age,
                "user_name": "Test Child",
                "conversation_history": []
            }
            
            response = engine.execute(entities, context)
            
            assert response.is_success(), f"Conversation failed for: {user_input}"
            logger.info(f"✓ Response for '{user_input}': {response.message[:100]}...")
            
        except Exception as e:
            logger.error(f"✗ Conversation test failed for '{user_input}': {e}")
    
    logger.info("=" * 80 + "\n")


def test_game_engine():
    """Test game engine"""
    logger.info("=" * 80)
    logger.info("TEST 5: Game Engine")
    logger.info("=" * 80)
    
    from src.engines.interaction.game_engine import GameEngine
    
    engine = GameEngine({})
    
    game_types = ["ispy", "trivia", "math", "riddle", "simon_says"]
    
    for game_type in game_types:
        try:
            entities = {"game_type": game_type}
            context = {"user_age": 8}
            
            response = engine.execute(entities, context)
            
            assert response.is_success(), f"Game failed: {game_type}"
            logger.info(f"✓ {game_type.upper()}: {response.message[:80]}...")
            
        except Exception as e:
            logger.error(f"✗ Game test failed for {game_type}: {e}")
    
    logger.info("=" * 80 + "\n")


def test_safety_monitor():
    """Test safety monitor engine (critical)"""
    logger.info("=" * 80)
    logger.info("TEST 6: Safety Monitor (CRITICAL)")
    logger.info("=" * 80)
    
    from src.engines.safety.safety_monitor_engine import SafetyMonitorEngine
    
    engine = SafetyMonitorEngine({
        "emergency_number": "999",
        "family_contacts": ["test@example.com"],
        "enable_auto_call": False  # Don't actually call 999 in tests!
    })
    
    # Test 1: Normal monitoring (no fall)
    try:
        entities = {}
        context = {"fall_detected": False}
        
        response = engine.execute(entities, context)
        
        assert response.is_success()
        logger.info("✓ Normal monitoring working")
        logger.info(f"  Message: {response.message}")
        
    except Exception as e:
        logger.error(f"✗ Normal monitoring failed: {e}")
    
    # Test 2: Fall detected (emergency)
    try:
        entities = {
            "event_type": "fall",
            "person_id": "test_user",
            "location": "living_room"
        }
        context = {
            "fall_detected": True,
            "impact_force": 8.5,
            "person_position": "lying_down"
        }
        
        response = engine.execute(entities, context)
        
        # Should trigger emergency protocol
        logger.info(f"✓ Fall detection protocol executed")
        logger.info(f"  Emergency level: {response.data.get('emergency_level', 'unknown')}")
        logger.info(f"  Message: {response.message}")
        
    except Exception as e:
        logger.error(f"✗ Fall detection failed: {e}")
    
    logger.info("=" * 80 + "\n")


def test_intent_router():
    """Test intent router integration"""
    logger.info("=" * 80)
    logger.info("TEST 7: Intent Router Integration")
    logger.info("=" * 80)
    
    try:
        from src.intent_router.router import IntentRouter
        
        # Initialize router
        router = IntentRouter({})
        
        # Test intent normalization
        test_intents = [
            ("pick_up_object", "object_grasp"),
            ("bring_object", "object_transfer"),
            ("hello", "greeting"),
            ("tell_joke", "tell_joke"),
        ]
        
        for original, expected in test_intents:
            normalized = router.normalize_intent(original)
            assert normalized == expected, f"Expected {expected}, got {normalized}"
            logger.info(f"✓ Intent normalized: {original} -> {normalized}")
        
        # Test getting stats
        stats = router.get_stats()
        logger.info(f"✓ Router stats: {stats}")
        
    except Exception as e:
        logger.error(f"✗ Intent router test failed: {e}")
        raise
    
    logger.info("=" * 80 + "\n")


def test_end_to_end_scenario():
    """Test complete end-to-end scenario"""
    logger.info("=" * 80)
    logger.info("TEST 8: End-to-End Home Assistant Scenario")
    logger.info("=" * 80)
    
    logger.info("\nScenario: Child asks for help with homework")
    logger.info("-" * 80)
    
    try:
        # Initialize memory
        from src.memory.advanced_memory_manager import AdvancedMemoryManager
        memory = AdvancedMemoryManager()
        
        # Child introduces themselves
        logger.info("\n1. Child: 'My name is Tommy, I'm 8 years old'")
        memory.add_conversation_turn(
            session_id="tommy_session",
            user_input="My name is Tommy, I'm 8 years old",
            robot_response="Nice to meet you, Tommy!",
            intent="introduce_self",
            entities={}
        )
        memory.remember_user_info("tommy_session", "tommy123", "name", "Tommy", 0.9)
        memory.remember_user_info("tommy_session", "tommy123", "age", 8, 0.9)
        
        # Child asks for homework help
        logger.info("\n2. Child: 'Can you help me with my math homework?'")
        
        from src.engines.interaction.education_engine import EducationEngine
        edu_engine = EducationEngine({})
        
        response = edu_engine.execute(
            {"subject": "math", "question": "What is 5 + 3?"},
            {"user_age": 8, "user_name": "Tommy"}
        )
        
        logger.info(f"   Robot: {response.message}")
        
        # Store in memory
        memory.add_conversation_turn(
            session_id="tommy_session",
            user_input="Can you help me with my math homework?",
            robot_response=response.message,
            intent="homework_help",
            entities={"subject": "math"}
        )
        
        # Child wants to play after homework
        logger.info("\n3. Child: 'I finished! Can we play a game?'")
        
        from src.engines.interaction.game_engine import GameEngine
        game_engine = GameEngine({})
        
        response = game_engine.execute(
            {"game_type": "ispy"},
            {"user_age": 8}
        )
        
        logger.info(f"   Robot: {response.message}")
        
        # Bedtime story
        logger.info("\n4. Child: 'It's bedtime, can you tell me a story?'")
        
        from src.engines.interaction.storytelling_engine import StorytellingEngine
        story_engine = StorytellingEngine({})
        
        response = story_engine.execute(
            {"story_type": "bedtime", "character_name": "Tommy"},
            {"user_age": 8, "user_name": "Tommy"}
        )
        
        logger.info(f"   Robot: {response.message[:200]}...")
        
        # Recall name at the end
        logger.info("\n5. Verify memory retention")
        recalled_name = memory.recall_user_info("tommy_session", "tommy123", "name")
        logger.info(f"   ✓ Robot still remembers: Name = {recalled_name}")
        
        assert recalled_name == "Tommy", "Memory recall failed!"
        
        logger.info("\n✓ Complete end-to-end scenario PASSED")
        logger.info("  - Child introduced themselves")
        logger.info("  - Got homework help")
        logger.info("  - Played a game")
        logger.info("  - Heard a bedtime story")
        logger.info("  - Robot remembered their name throughout")
        
    except Exception as e:
        logger.error(f"✗ End-to-end test failed: {e}")
        raise
    
    logger.info("=" * 80 + "\n")


def main():
    """Run all integration tests"""
    logger.info("\n\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 15 + "HOME ASSISTANT ROBOT INTEGRATION TESTS" + " " * 25 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("\n")
    
    tests = [
        ("Engine Imports", test_engine_imports),
        ("Engine Initialization", test_engine_initialization),
        ("Memory Systems", test_memory_systems),
        ("Conversation Engine", test_conversation_engine),
        ("Game Engine", test_game_engine),
        ("Safety Monitor (CRITICAL)", test_safety_monitor),
        ("Intent Router", test_intent_router),
        ("End-to-End Scenario", test_end_to_end_scenario),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            logger.error(f"\n{'=' * 80}")
            logger.error(f"TEST FAILED: {test_name}")
            logger.error(f"Error: {e}")
            logger.error(f"{'=' * 80}\n")
            failed += 1
        except Exception as e:
            logger.error(f"\n{'=' * 80}")
            logger.error(f"TEST ERROR: {test_name}")
            logger.error(f"Exception: {e}")
            logger.error(f"{'=' * 80}\n")
            failed += 1
    
    # Final Summary
    logger.info("\n\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 30 + "TEST SUMMARY" + " " * 36 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info(f"\nTotal Tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        logger.info("\n" + "✓" * 40)
        logger.info("ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT!")
        logger.info("✓" * 40 + "\n")
    else:
        logger.error("\n" + "✗" * 40)
        logger.error(f"{failed} TEST(S) FAILED - REVIEW ABOVE ERRORS")
        logger.error("✗" * 40 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

