"""
Dialogue Manager with Multi-Tier Fallback
Tier 1: Custom State Machine + Redis
Tier 2: Rasa (fallback)
Tier 3: LangChain ConversationBufferMemory

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta

# Redis for session management
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Install with: pip install redis")

# LangChain for context management
try:
    from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Install with: pip install langchain")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DialogueState(Enum):
    """Dialogue states for the robot conversation"""
    IDLE = "idle"
    LISTENING = "listening"
    UNDERSTANDING = "understanding"
    CLARIFYING = "clarifying"
    EXECUTING = "executing"
    RESPONDING = "responding"
    ERROR = "error"


@dataclass
class DialogueTurn:
    """Represents a single turn in the dialogue"""
    user_input: str
    intent: Optional[str]
    entities: Dict[str, Any]
    response: Optional[str]
    timestamp: float
    state: str


@dataclass
class DialogueSession:
    """Represents a dialogue session"""
    session_id: str
    user_id: str
    start_time: float
    last_activity: float
    state: str
    context: Dict[str, Any]
    turns: List[DialogueTurn]
    slots: Dict[str, Any]  # Slot filling


class DialogueManager:
    """
    Multi-tier dialogue management with automatic fallback.
    Tier 1: State Machine + Redis
    Tier 2: LangChain Memory
    Tier 3: In-memory fallback
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        session_ttl: int = 1800,  # 30 minutes
        context_window: int = 10,  # Last N turns
        max_clarifications: int = 3
    ):
        """
        Initialize dialogue manager.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_password: Redis password (if any)
            session_ttl: Session time-to-live in seconds
            context_window: Number of recent turns to keep
            max_clarifications: Maximum clarification attempts
        """
        self.session_ttl = session_ttl
        self.context_window = context_window
        self.max_clarifications = max_clarifications
        
        # Initialize storage tiers
        self.redis_client = None
        self.langchain_memory = {}  # session_id -> memory
        self.memory_store = {}  # In-memory fallback
        
        self._initialize_redis(redis_host, redis_port, redis_password)
        self._initialize_langchain()
        
        # State transitions
        self.state_transitions = {
            DialogueState.IDLE: [DialogueState.LISTENING],
            DialogueState.LISTENING: [DialogueState.UNDERSTANDING, DialogueState.ERROR],
            DialogueState.UNDERSTANDING: [DialogueState.CLARIFYING, DialogueState.EXECUTING, DialogueState.ERROR],
            DialogueState.CLARIFYING: [DialogueState.UNDERSTANDING, DialogueState.EXECUTING, DialogueState.ERROR],
            DialogueState.EXECUTING: [DialogueState.RESPONDING, DialogueState.ERROR],
            DialogueState.RESPONDING: [DialogueState.IDLE, DialogueState.LISTENING],
            DialogueState.ERROR: [DialogueState.IDLE],
        }
    
    def _initialize_redis(self, host: str, port: int, password: Optional[str]):
        """Initialize Redis connection (Tier 1)"""
        if not REDIS_AVAILABLE:
            logger.warning("Tier 1 (Redis) unavailable: redis not installed")
            return
        
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                password=password,
                decode_responses=True,
                socket_connect_timeout=2
            )
            # Test connection
            self.redis_client.ping()
            logger.info("✓ Tier 1 (Redis) initialized successfully")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis connection failed: {e}. Using fallback storage.")
            self.redis_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None
    
    def _initialize_langchain(self):
        """Initialize LangChain memory (Tier 2)"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("Tier 2 (LangChain) unavailable: langchain not installed")
            return
        
        logger.info("✓ Tier 2 (LangChain) initialized successfully")
    
    def create_session(
        self,
        session_id: str,
        user_id: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> DialogueSession:
        """
        Create a new dialogue session.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            initial_context: Initial context data
            
        Returns:
            New dialogue session
        """
        session = DialogueSession(
            session_id=session_id,
            user_id=user_id,
            start_time=time.time(),
            last_activity=time.time(),
            state=DialogueState.IDLE.value,
            context=initial_context or {},
            turns=[],
            slots={}
        )
        
        self._save_session(session)
        
        # Initialize LangChain memory for this session
        if LANGCHAIN_AVAILABLE:
            self.langchain_memory[session_id] = ConversationBufferWindowMemory(
                k=self.context_window,
                return_messages=True
            )
        
        logger.info(f"Created new session: {session_id} for user: {user_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[DialogueSession]:
        """
        Retrieve a dialogue session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dialogue session or None if not found
        """
        # Try Tier 1: Redis
        if self.redis_client is not None:
            try:
                session_data = self.redis_client.get(f"session:{session_id}")
                if session_data:
                    data = json.loads(session_data)
                    return self._dict_to_session(data)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}. Trying fallback...")
        
        # Try Tier 3: In-memory
        if session_id in self.memory_store:
            return self.memory_store[session_id]
        
        return None
    
    def update_session(
        self,
        session_id: str,
        user_input: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        response: Optional[str] = None,
        new_state: Optional[DialogueState] = None
    ) -> DialogueSession:
        """
        Update dialogue session with new turn.
        
        Args:
            session_id: Session identifier
            user_input: User's input text
            intent: Detected intent
            entities: Extracted entities
            response: System response
            new_state: New dialogue state
            
        Returns:
            Updated session
        """
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        
        # Create new turn
        turn = DialogueTurn(
            user_input=user_input,
            intent=intent,
            entities=entities or {},
            response=response,
            timestamp=time.time(),
            state=session.state
        )
        
        # Update session
        session.turns.append(turn)
        session.last_activity = time.time()
        
        # Update slots with extracted entities
        if entities:
            session.slots.update(entities)
        
        # Update state if provided
        if new_state:
            if self._is_valid_transition(DialogueState(session.state), new_state):
                session.state = new_state.value
            else:
                logger.warning(f"Invalid state transition: {session.state} -> {new_state.value}")
        
        # Keep only last N turns
        if len(session.turns) > self.context_window:
            session.turns = session.turns[-self.context_window:]
        
        # Update LangChain memory
        if LANGCHAIN_AVAILABLE and session_id in self.langchain_memory:
            try:
                self.langchain_memory[session_id].save_context(
                    {"input": user_input},
                    {"output": response or ""}
                )
            except Exception as e:
                logger.warning(f"LangChain memory update failed: {e}")
        
        # Save session
        self._save_session(session)
        
        return session
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get conversation context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Context dictionary
        """
        session = self.get_session(session_id)
        if session is None:
            return {}
        
        # Build context from recent turns
        context = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "current_state": session.state,
            "slots": session.slots,
            "recent_intents": [t.intent for t in session.turns[-5:] if t.intent],
            "turn_count": len(session.turns),
            "context_data": session.context
        }
        
        # Add LangChain memory if available
        if LANGCHAIN_AVAILABLE and session_id in self.langchain_memory:
            try:
                context["langchain_history"] = self.langchain_memory[session_id].load_memory_variables({})
            except Exception as e:
                logger.warning(f"Failed to load LangChain memory: {e}")
        
        return context
    
    def needs_clarification(self, session_id: str, missing_slots: List[str]) -> bool:
        """
        Check if clarification is needed.
        
        Args:
            session_id: Session identifier
            missing_slots: List of missing required slots
            
        Returns:
            True if clarification is needed
        """
        session = self.get_session(session_id)
        if session is None or not missing_slots:
            return False
        
        # Count recent clarifications
        recent_clarifications = sum(
            1 for turn in session.turns[-self.max_clarifications:]
            if turn.state == DialogueState.CLARIFYING.value
        )
        
        return recent_clarifications < self.max_clarifications
    
    def delete_session(self, session_id: str):
        """Delete a dialogue session"""
        # Remove from Redis
        if self.redis_client is not None:
            try:
                self.redis_client.delete(f"session:{session_id}")
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
        
        # Remove from in-memory store
        self.memory_store.pop(session_id, None)
        
        # Remove from LangChain memory
        self.langchain_memory.pop(session_id, None)
        
        logger.info(f"Deleted session: {session_id}")
    
    def _save_session(self, session: DialogueSession):
        """Save session to storage"""
        # Try Tier 1: Redis
        if self.redis_client is not None:
            try:
                session_data = self._session_to_dict(session)
                self.redis_client.setex(
                    f"session:{session.session_id}",
                    self.session_ttl,
                    json.dumps(session_data)
                )
                return
            except Exception as e:
                logger.warning(f"Redis save failed: {e}. Using fallback...")
        
        # Tier 3: In-memory fallback
        self.memory_store[session.session_id] = session
    
    def _is_valid_transition(self, from_state: DialogueState, to_state: DialogueState) -> bool:
        """Check if state transition is valid"""
        return to_state in self.state_transitions.get(from_state, [])
    
    def _session_to_dict(self, session: DialogueSession) -> Dict:
        """Convert session to dictionary"""
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "start_time": session.start_time,
            "last_activity": session.last_activity,
            "state": session.state,
            "context": session.context,
            "turns": [asdict(turn) for turn in session.turns],
            "slots": session.slots
        }
    
    def _dict_to_session(self, data: Dict) -> DialogueSession:
        """Convert dictionary to session"""
        turns = [DialogueTurn(**turn_data) for turn_data in data.get("turns", [])]
        return DialogueSession(
            session_id=data["session_id"],
            user_id=data["user_id"],
            start_time=data["start_time"],
            last_activity=data["last_activity"],
            state=data["state"],
            context=data.get("context", {}),
            turns=turns,
            slots=data.get("slots", {})
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get dialogue manager status"""
        return {
            "tier1_redis": self.redis_client is not None,
            "tier2_langchain": LANGCHAIN_AVAILABLE,
            "tier3_memory": True,  # Always available
            "active_sessions": len(self.memory_store) if self.redis_client is None else "redis",
        }


# Example usage
if __name__ == "__main__":
    # Initialize dialogue manager
    manager = DialogueManager()
    
    print("=" * 80)
    print("DIALOGUE MANAGER - TESTING")
    print("=" * 80)
    print(f"\nStatus: {manager.get_status()}\n")
    
    # Create session
    session = manager.create_session("test_session_001", "user_123")
    print(f"Created session: {session.session_id}")
    
    # Simulate conversation
    conversations = [
        ("bring me the red cup", "fetch_object", {"object": "cup", "color": "red"}),
        ("from the kitchen", "location", {"location": "kitchen"}),
        ("yes please", "confirm", {}),
    ]
    
    for user_input, intent, entities in conversations:
        response = f"Processing: {intent}"
        session = manager.update_session(
            session.session_id,
            user_input=user_input,
            intent=intent,
            entities=entities,
            response=response,
            new_state=DialogueState.RESPONDING
        )
        print(f"\nUser: {user_input}")
        print(f"Intent: {intent} | Entities: {entities}")
        print(f"Slots filled: {session.slots}")
    
    # Get context
    context = manager.get_context(session.session_id)
    print(f"\nFinal Context: {json.dumps(context, indent=2)}")
    
    print("\n" + "=" * 80)

