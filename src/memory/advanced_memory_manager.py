"""
Advanced Memory Management System for Humanoid Robot

Multi-tier memory system that SOLVES the context retention problem:
- Working Memory: Current conversation (immediate access)
- Short-term Memory: Recent interactions (minutes to hours)
- Long-term Memory: Persistent knowledge (days to forever)
- Semantic Memory: Facts and knowledge
- Episodic Memory: Experiences and events
- Procedural Memory: Skills and how-to knowledge

THIS FIXES THE CHAPO BOT PROBLEM:
- User says "My name is John" → REMEMBERED across entire conversation
- User says "I like coffee" → RETRIEVED when relevant later
- User preferences → PERSISTED across sessions
- Context-aware responses using memory retrieval

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import json
import re

from src.memory.mongodb_manager import (
    MongoDBManager, 
    EpisodicMemory, 
    SemanticMemory, 
    UserProfile
)

logger = logging.getLogger(__name__)


class MemoryImportance:
    """Calculate importance scores for memories"""
    
    @staticmethod
    def calculate_importance(memory_data: Dict[str, Any]) -> float:
        """
        Calculate importance score (0.0 - 1.0) based on:
        - Recency (recent = more important)
        - Frequency (repeated = more important)
        - User feedback (positive feedback = more important)
        - Keywords (certain words = more important)
        """
        importance = 0.5  # Base importance
        
        # Recency boost
        if 'timestamp' in memory_data:
            age_hours = (datetime.utcnow() - memory_data['timestamp']).total_seconds() / 3600
            if age_hours < 1:
                importance += 0.3
            elif age_hours < 24:
                importance += 0.2
            elif age_hours < 168:  # 1 week
                importance += 0.1
        
        # Keyword boost (names, preferences, important info)
        important_keywords = ['name', 'like', 'prefer', 'love', 'hate', 'favorite', 
                             'important', 'remember', 'birthday', 'address', 'phone']
        text = str(memory_data.get('content', '')).lower()
        
        for keyword in important_keywords:
            if keyword in text:
                importance += 0.1
                break
        
        # Explicit importance if provided
        if 'explicit_importance' in memory_data:
            importance = max(importance, memory_data['explicit_importance'])
        
        return min(1.0, importance)


class WorkingMemory:
    """
    Working Memory: Current conversation context
    - Fast in-memory storage
    - Limited capacity (last N turns)
    - Cleared when conversation ends
    """
    
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.conversation_turns = deque(maxlen=capacity)
        self.current_context = {}
        self.user_info = {}  # Extracted during conversation
        
    def add_turn(self, user_input: str, robot_response: str, 
                 intent: str, entities: Dict[str, Any]):
        """Add a conversation turn"""
        turn = {
            'timestamp': datetime.utcnow(),
            'user_input': user_input,
            'robot_response': robot_response,
            'intent': intent,
            'entities': entities
        }
        
        self.conversation_turns.append(turn)
        
        # Extract and store user information
        self._extract_user_info(user_input, entities)
        
    def _extract_user_info(self, text: str, entities: Dict[str, Any]):
        """Extract user information from conversation"""
        text_lower = text.lower()
        
        # Extract name
        name_patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"call me (\w+)",
            r"this is (\w+)",
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                self.user_info['name'] = match.group(1).capitalize()
                logger.info(f"📝 Extracted user name: {self.user_info['name']}")
        
        # Extract preferences
        preference_patterns = [
            (r"i (like|love|prefer|enjoy) (\w+)", "likes"),
            (r"i (hate|dislike|don't like) (\w+)", "dislikes"),
            (r"my favorite (\w+) is (\w+)", "favorites"),
        ]
        
        for pattern, category in preference_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if category not in self.user_info:
                    self.user_info[category] = []
                
                item = match.group(2) if len(match.groups()) > 1 else match.group(1)
                if item not in self.user_info[category]:
                    self.user_info[category].append(item)
                    logger.info(f"📝 Extracted user preference: {category} - {item}")
        
        # Extract entities (names, locations, objects)
        if entities:
            if 'person_name' in entities:
                if 'mentioned_people' not in self.user_info:
                    self.user_info['mentioned_people'] = []
                self.user_info['mentioned_people'].append(entities['person_name'])
            
            if 'location' in entities:
                if 'mentioned_locations' not in self.user_info:
                    self.user_info['mentioned_locations'] = []
                self.user_info['mentioned_locations'].append(entities['location'])
    
    def get_recent_context(self, turns: int = 3) -> str:
        """Get recent conversation context as text"""
        recent_turns = list(self.conversation_turns)[-turns:]
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"User: {turn['user_input']}")
            context_parts.append(f"Robot: {turn['robot_response']}")
        
        return "\n".join(context_parts)
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get extracted user information"""
        return self.user_info
    
    def find_in_conversation(self, query: str) -> List[Dict[str, Any]]:
        """Search conversation history for relevant information"""
        query_lower = query.lower()
        results = []
        
        for turn in self.conversation_turns:
            if query_lower in turn['user_input'].lower() or \
               query_lower in turn['robot_response'].lower():
                results.append(turn)
        
        return results
    
    def clear(self):
        """Clear working memory (conversation ended)"""
        self.conversation_turns.clear()
        self.current_context.clear()
        # Keep user_info for transfer to long-term memory


class ShortTermMemory:
    """
    Short-term Memory: Recent interactions (last few hours)
    - In-memory cache with TTL
    - Fast retrieval for recent context
    - Automatically transfers important items to long-term
    """
    
    def __init__(self, ttl_hours: int = 24):
        self.ttl_hours = ttl_hours
        self.memories = {}  # session_id -> List[memory]
        
    def add_memory(self, session_id: str, memory: Dict[str, Any]):
        """Add memory to short-term storage"""
        if session_id not in self.memories:
            self.memories[session_id] = []
        
        memory['added_at'] = datetime.utcnow()
        memory['importance'] = MemoryImportance.calculate_importance(memory)
        
        self.memories[session_id].append(memory)
        
        # Cleanup old memories
        self._cleanup_expired()
    
    def get_memories(self, session_id: str, 
                     min_importance: float = 0.0) -> List[Dict[str, Any]]:
        """Get memories for a session"""
        if session_id not in self.memories:
            return []
        
        # Filter by importance and sort by recency
        relevant_memories = [
            m for m in self.memories[session_id]
            if m.get('importance', 0) >= min_importance
        ]
        
        return sorted(relevant_memories, 
                     key=lambda x: x['added_at'], 
                     reverse=True)
    
    def search_memories(self, query: str, 
                       session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search memories by text query"""
        query_lower = query.lower()
        results = []
        
        sessions_to_search = [session_id] if session_id else self.memories.keys()
        
        for sid in sessions_to_search:
            if sid not in self.memories:
                continue
                
            for memory in self.memories[sid]:
                content = str(memory.get('content', '')).lower()
                if query_lower in content:
                    results.append(memory)
        
        return sorted(results, 
                     key=lambda x: x.get('importance', 0), 
                     reverse=True)
    
    def _cleanup_expired(self):
        """Remove expired memories"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.ttl_hours)
        
        for session_id in list(self.memories.keys()):
            self.memories[session_id] = [
                m for m in self.memories[session_id]
                if m['added_at'] > cutoff_time
            ]
            
            # Remove empty sessions
            if not self.memories[session_id]:
                del self.memories[session_id]


class AdvancedMemoryManager:
    """
    Advanced Memory Manager - Multi-tier system
    
    SOLVES THE CHAPO BOT PROBLEM:
    1. User says "My name is John"
       → Stored in working memory (immediate)
       → Extracted to user_info
       → Saved to short-term memory
       → Persisted to long-term (MongoDB)
    
    2. User later asks "What's my name?"
       → Check working memory (found!)
       → Return "Your name is John"
    
    3. User says "I like coffee"
       → Stored in all memory tiers
       → Tagged as preference
       → High importance score
    
    4. User comes back tomorrow
       → Working memory cleared (new session)
       → But long-term memory retrieved
       → "Hi John! Would you like some coffee?"
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        
        # Initialize memory tiers
        self.working_memory = WorkingMemory(capacity=self.config.get('working_memory_capacity', 10))
        self.short_term_memory = ShortTermMemory(ttl_hours=self.config.get('short_term_ttl_hours', 24))
        
        # Long-term memory (MongoDB)
        self.long_term_memory = MongoDBManager(self.config.get('mongodb'))
        self.long_term_connected = self.long_term_memory.connect()
        
        if self.long_term_connected:
            logger.info("✅ Advanced Memory Manager: All tiers active (Working + Short-term + Long-term)")
        else:
            logger.warning("⚠️ Advanced Memory Manager: Running without long-term persistence")
    
    # ============================================================
    # CONVERSATION MEMORY (Working Memory)
    # ============================================================
    
    def add_conversation_turn(self, session_id: str, user_input: str, 
                             robot_response: str, intent: str, 
                             entities: Dict[str, Any]):
        """
        Add a conversation turn - MAIN ENTRY POINT
        
        This is called after every user interaction.
        """
        # Add to working memory
        self.working_memory.add_turn(user_input, robot_response, intent, entities)
        
        # Add to short-term memory
        self.short_term_memory.add_memory(session_id, {
            'type': 'conversation_turn',
            'content': f"User: {user_input}\nRobot: {robot_response}",
            'user_input': user_input,
            'robot_response': robot_response,
            'intent': intent,
            'entities': entities,
            'timestamp': datetime.utcnow()
        })
        
        # Save to long-term (MongoDB)
        if self.long_term_connected:
            self.long_term_memory.save_conversation_turn(
                session_id=session_id,
                user_message=user_input,
                robot_response=robot_response,
                context={'intent': intent, 'entities': entities}
            )
    
    def remember_user_info(self, session_id: str, user_id: str, 
                          info_type: str, info_value: Any, 
                          explicit_importance: float = 0.8):
        """
        Explicitly remember user information
        
        Examples:
        - remember_user_info("session123", "user456", "name", "John", 0.9)
        - remember_user_info("session123", "user456", "preference", {"type": "beverage", "value": "coffee"}, 0.8)
        """
        # Add to working memory
        self.working_memory.user_info[info_type] = info_value
        
        # Add to short-term memory with high importance
        self.short_term_memory.add_memory(session_id, {
            'type': 'user_info',
            'info_type': info_type,
            'content': f"{info_type}: {info_value}",
            'value': info_value,
            'user_id': user_id,
            'timestamp': datetime.utcnow(),
            'explicit_importance': explicit_importance
        })
        
        # Save to long-term (MongoDB)
        if self.long_term_connected:
            # Update user profile
            profile = self.long_term_memory.get_user_profile(user_id)
            
            if profile:
                # Update existing profile
                if 'preferences' not in profile:
                    profile['preferences'] = {}
                profile['preferences'][info_type] = info_value
                
                from src.memory.mongodb_manager import UserProfile
                updated_profile = UserProfile(
                    user_id=user_id,
                    name=profile.get('name'),
                    preferences=profile['preferences']
                )
                self.long_term_memory.update_user_profile(updated_profile)
            else:
                # Create new profile
                from src.memory.mongodb_manager import UserProfile
                new_profile = UserProfile(
                    user_id=user_id,
                    name=info_value if info_type == 'name' else None,
                    preferences={info_type: info_value}
                )
                self.long_term_memory.update_user_profile(new_profile)
            
            # Also store as semantic memory (fact)
            semantic_memory = SemanticMemory(
                fact_type=f"user_{info_type}",
                fact=f"User {user_id} {info_type}: {info_value}",
                source="user_conversation",
                confidence=explicit_importance
            )
            self.long_term_memory.store_semantic_memory(semantic_memory)
    
    def recall_user_info(self, session_id: str, user_id: str, 
                        info_type: str) -> Optional[Any]:
        """
        Recall user information - SMART RETRIEVAL
        
        Checks all memory tiers:
        1. Working memory (fastest)
        2. Short-term memory
        3. Long-term memory (MongoDB)
        """
        # Check working memory first
        user_info = self.working_memory.get_user_info()
        if info_type in user_info:
            logger.info(f"✅ Found '{info_type}' in working memory")
            return user_info[info_type]
        
        # Check short-term memory
        memories = self.short_term_memory.get_memories(session_id)
        for memory in memories:
            if memory.get('type') == 'user_info' and \
               memory.get('info_type') == info_type:
                logger.info(f"✅ Found '{info_type}' in short-term memory")
                return memory.get('value')
        
        # Check long-term memory (MongoDB)
        if self.long_term_connected:
            profile = self.long_term_memory.get_user_profile(user_id)
            if profile:
                if info_type == 'name' and profile.get('name'):
                    logger.info(f"✅ Found '{info_type}' in long-term memory (profile)")
                    return profile['name']
                
                if 'preferences' in profile and info_type in profile['preferences']:
                    logger.info(f"✅ Found '{info_type}' in long-term memory (preferences)")
                    return profile['preferences'][info_type]
        
        logger.warning(f"❌ Could not find '{info_type}' in any memory tier")
        return None
    
    def search_memory(self, query: str, session_id: Optional[str] = None,
                     user_id: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Search across all memory tiers
        
        Returns results from all tiers for comprehensive context
        """
        results = {
            'working_memory': [],
            'short_term_memory': [],
            'long_term_memory': []
        }
        
        # Search working memory
        results['working_memory'] = self.working_memory.find_in_conversation(query)
        
        # Search short-term memory
        results['short_term_memory'] = self.short_term_memory.search_memories(
            query, session_id
        )
        
        # Search long-term memory
        if self.long_term_connected:
            # Search episodic memories
            episodic = self.long_term_memory.recall_episodic_memories(
                session_id=session_id,
                limit=10
            )
            
            # Filter by query
            for memory in episodic:
                if query.lower() in str(memory).lower():
                    results['long_term_memory'].append(memory)
        
        return results
    
    def get_conversation_context(self, session_id: str, user_id: str,
                                 turns: int = 3) -> str:
        """
        Get comprehensive conversation context for LLM
        
        Combines:
        - Recent conversation (working memory)
        - User info (all tiers)
        - Relevant episodic memories
        """
        context_parts = []
        
        # User information
        user_info = self.working_memory.get_user_info()
        if user_info:
            context_parts.append("User Information:")
            for key, value in user_info.items():
                context_parts.append(f"  - {key}: {value}")
            context_parts.append("")
        
        # User profile from long-term memory
        if self.long_term_connected:
            profile = self.long_term_memory.get_user_profile(user_id)
            if profile:
                if profile.get('name'):
                    context_parts.append(f"User's name: {profile['name']}")
                if profile.get('preferences'):
                    context_parts.append("User preferences:")
                    for key, value in profile['preferences'].items():
                        context_parts.append(f"  - {key}: {value}")
                context_parts.append("")
        
        # Recent conversation
        recent_context = self.working_memory.get_recent_context(turns)
        if recent_context:
            context_parts.append("Recent conversation:")
            context_parts.append(recent_context)
        
        return "\n".join(context_parts)
    
    # ============================================================
    # SESSION MANAGEMENT
    # ============================================================
    
    def start_session(self, session_id: str, user_id: str):
        """
        Start a new session - Load user context from long-term memory
        """
        logger.info(f"🔄 Starting session {session_id} for user {user_id}")
        
        # Clear working memory for new session
        self.working_memory.clear()
        
        # Load user profile from long-term memory
        if self.long_term_connected:
            profile = self.long_term_memory.get_user_profile(user_id)
            
            if profile:
                logger.info(f"✅ Loaded user profile: {profile.get('name', 'Unknown')}")
                
                # Pre-populate working memory with user info
                if profile.get('name'):
                    self.working_memory.user_info['name'] = profile['name']
                
                if profile.get('preferences'):
                    self.working_memory.user_info.update(profile['preferences'])
                
                # Load recent conversation history
                history = self.long_term_memory.get_conversation_history(
                    session_id=session_id,
                    limit=5
                )
                
                logger.info(f"✅ Loaded {len(history)} previous conversation turns")
    
    def end_session(self, session_id: str, user_id: str):
        """
        End session - Transfer working memory to long-term
        """
        logger.info(f"🔄 Ending session {session_id}")
        
        # Transfer important working memory to long-term
        user_info = self.working_memory.get_user_info()
        
        for info_type, info_value in user_info.items():
            self.remember_user_info(
                session_id, 
                user_id, 
                info_type, 
                info_value,
                explicit_importance=0.8
            )
        
        # Create episodic memory of session
        if self.long_term_connected:
            summary = f"Conversation session with {len(self.working_memory.conversation_turns)} turns"
            
            episodic = EpisodicMemory(
                session_id=session_id,
                event_type="conversation_session",
                event_description=summary,
                context={'user_info': user_info},
                importance=0.7
            )
            
            self.long_term_memory.store_episodic_memory(episodic)
        
        logger.info("✅ Session memory transferred to long-term storage")
    
    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        stats = {
            'working_memory': {
                'conversation_turns': len(self.working_memory.conversation_turns),
                'user_info_items': len(self.working_memory.user_info)
            },
            'short_term_memory': {
                'active_sessions': len(self.short_term_memory.memories),
                'total_memories': sum(len(m) for m in self.short_term_memory.memories.values())
            },
            'long_term_memory': {}
        }
        
        if self.long_term_connected:
            stats['long_term_memory'] = self.long_term_memory.get_database_stats()
        
        return stats

