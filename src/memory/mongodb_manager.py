"""
MongoDB Memory and Logging Manager for Humanoid Robot

Comprehensive memory and logging system using MongoDB for:
- Interaction logging
- Episodic memory
- Semantic memory  
- Performance metrics
- User profiles
- Conversation history
- Learning feedback

Based on Chapo bot's successful MongoDB implementation.

Author: Victor Ibhafidon
Date: October 2025
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from urllib.parse import quote_plus
from pymongo import MongoClient, DESCENDING, ASCENDING
from pymongo.errors import PyMongoError
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class InteractionLog:
    """Structure for interaction logs"""
    session_id: str
    user_input: str
    intent: str
    normalized_intent: str
    entities: Dict[str, Any]
    engine_responses: List[Dict[str, Any]]
    final_response: str
    execution_time: float
    success: bool
    timestamp: datetime = None
    user_id: Optional[str] = None
    emotion: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class EpisodicMemory:
    """Structure for episodic memories"""
    session_id: str
    event_type: str
    event_description: str
    context: Dict[str, Any]
    timestamp: datetime = None
    importance: float = 0.5
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class SemanticMemory:
    """Structure for semantic knowledge"""
    fact_type: str
    fact: str
    source: str
    confidence: float
    related_facts: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.related_facts is None:
            self.related_facts = []


@dataclass
class UserProfile:
    """Structure for user profiles"""
    user_id: str
    name: Optional[str] = None
    preferences: Dict[str, Any] = None
    interaction_count: int = 0
    first_seen: datetime = None
    last_seen: datetime = None
    
    def __post_init__(self):
        if self.first_seen is None:
            self.first_seen = datetime.utcnow()
        if self.last_seen is None:
            self.last_seen = datetime.utcnow()
        if self.preferences is None:
            self.preferences = {}


@dataclass
class PerformanceMetric:
    """Structure for performance metrics"""
    metric_type: str
    metric_name: str
    value: float
    unit: str
    component: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class MongoDBManager:
    """
    MongoDB Manager for robot memory and logging
    
    Collections:
    - interactions: All user interactions
    - episodic_memory: Experience-based memories
    - semantic_memory: Fact-based knowledge
    - user_profiles: User information and preferences
    - performance_metrics: System performance data
    - feedback_logs: User feedback for learning
    - conversation_history: Detailed conversation logs
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        self.client = None
        self.db = None
        self.connected = False
        
        # Collection names
        self.INTERACTIONS = "interactions"
        self.EPISODIC_MEMORY = "episodic_memory"
        self.SEMANTIC_MEMORY = "semantic_memory"
        self.USER_PROFILES = "user_profiles"
        self.PERFORMANCE_METRICS = "performance_metrics"
        self.FEEDBACK_LOGS = "feedback_logs"
        self.CONVERSATION_HISTORY = "conversation_history"
        
    def connect(self) -> bool:
        """
        Connect to MongoDB using environment variables
        
        Required environment variables:
        - MONGODB_USERNAME
        - MONGODB_PASSWORD
        - MONGODB_DATABASE (default: robot_db)
        - MONGODB_CLUSTER (optional)
        """
        try:
            username = os.getenv("MONGODB_USERNAME")
            password = os.getenv("MONGODB_PASSWORD")
            database = os.getenv("MONGODB_DATABASE", "robot_db")
            cluster = os.getenv("MONGODB_CLUSTER", "cluster0.mongodb.net")
            
            if not username or not password:
                logger.warning("MongoDB credentials not found. Running without persistent storage.")
                return False
            
            # Encode credentials
            encoded_username = quote_plus(username)
            encoded_password = quote_plus(password)
            
            # Build MongoDB URI
            mongo_uri = (
                f"mongodb+srv://{encoded_username}:{encoded_password}@"
                f"{cluster}/{database}"
                "?retryWrites=true&w=majority"
            )
            
            # Connect
            self.client = MongoClient(mongo_uri, tls=True, serverSelectionTimeoutMS=5000)
            self.db = self.client[database]
            
            # Test connection
            self.client.server_info()
            
            self.connected = True
            logger.info(f"✅ MongoDB connected to database: {database}")
            
            # Create indexes for performance
            self._create_indexes()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.connected = False
            return False
    
    def _create_indexes(self):
        """Create indexes for better query performance"""
        try:
            # Interactions indexes
            self.db[self.INTERACTIONS].create_index([("session_id", ASCENDING)])
            self.db[self.INTERACTIONS].create_index([("timestamp", DESCENDING)])
            self.db[self.INTERACTIONS].create_index([("intent", ASCENDING)])
            
            # Episodic memory indexes
            self.db[self.EPISODIC_MEMORY].create_index([("session_id", ASCENDING)])
            self.db[self.EPISODIC_MEMORY].create_index([("timestamp", DESCENDING)])
            self.db[self.EPISODIC_MEMORY].create_index([("importance", DESCENDING)])
            
            # Semantic memory indexes
            self.db[self.SEMANTIC_MEMORY].create_index([("fact_type", ASCENDING)])
            self.db[self.SEMANTIC_MEMORY].create_index([("confidence", DESCENDING)])
            
            # User profiles indexes
            self.db[self.USER_PROFILES].create_index([("user_id", ASCENDING)], unique=True)
            
            # Performance metrics indexes
            self.db[self.PERFORMANCE_METRICS].create_index([("component", ASCENDING)])
            self.db[self.PERFORMANCE_METRICS].create_index([("timestamp", DESCENDING)])
            
            logger.info("✅ MongoDB indexes created")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create indexes: {e}")
    
    # ============================================================
    # INTERACTION LOGGING
    # ============================================================
    
    def log_interaction(self, interaction: InteractionLog) -> bool:
        """Log a user interaction"""
        if not self.connected:
            logger.debug("MongoDB not connected, skipping interaction log")
            return False
        
        try:
            data = asdict(interaction)
            self.db[self.INTERACTIONS].insert_one(data)
            logger.debug(f"📝 Interaction logged: {interaction.intent}")
            return True
        except PyMongoError as e:
            logger.error(f"❌ Failed to log interaction: {e}")
            return False
    
    def get_interactions(self, session_id: Optional[str] = None, 
                        limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interactions"""
        if not self.connected:
            return []
        
        try:
            query = {}
            if session_id:
                query["session_id"] = session_id
            
            interactions = self.db[self.INTERACTIONS].find(query).sort(
                "timestamp", DESCENDING
            ).limit(limit)
            
            return list(interactions)
        except PyMongoError as e:
            logger.error(f"❌ Failed to retrieve interactions: {e}")
            return []
    
    def get_interaction_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get interaction statistics for the past N days"""
        if not self.connected:
            return {}
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            pipeline = [
                {"$match": {"timestamp": {"$gte": cutoff_date}}},
                {"$group": {
                    "_id": "$intent",
                    "count": {"$sum": 1},
                    "success_rate": {"$avg": {"$cond": ["$success", 1, 0]}},
                    "avg_execution_time": {"$avg": "$execution_time"}
                }},
                {"$sort": {"count": -1}}
            ]
            
            results = list(self.db[self.INTERACTIONS].aggregate(pipeline))
            
            total_interactions = sum(r["count"] for r in results)
            
            return {
                "total_interactions": total_interactions,
                "unique_intents": len(results),
                "intent_breakdown": results,
                "period_days": days
            }
        except PyMongoError as e:
            logger.error(f"❌ Failed to get interaction stats: {e}")
            return {}
    
    # ============================================================
    # EPISODIC MEMORY
    # ============================================================
    
    def store_episodic_memory(self, memory: EpisodicMemory) -> bool:
        """Store an episodic memory"""
        if not self.connected:
            return False
        
        try:
            data = asdict(memory)
            self.db[self.EPISODIC_MEMORY].insert_one(data)
            logger.debug(f"🧠 Episodic memory stored: {memory.event_type}")
            return True
        except PyMongoError as e:
            logger.error(f"❌ Failed to store episodic memory: {e}")
            return False
    
    def recall_episodic_memories(self, session_id: Optional[str] = None,
                                 event_type: Optional[str] = None,
                                 min_importance: float = 0.0,
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """Recall episodic memories"""
        if not self.connected:
            return []
        
        try:
            query = {"importance": {"$gte": min_importance}}
            if session_id:
                query["session_id"] = session_id
            if event_type:
                query["event_type"] = event_type
            
            memories = self.db[self.EPISODIC_MEMORY].find(query).sort(
                [("importance", DESCENDING), ("timestamp", DESCENDING)]
            ).limit(limit)
            
            return list(memories)
        except PyMongoError as e:
            logger.error(f"❌ Failed to recall episodic memories: {e}")
            return []
    
    # ============================================================
    # SEMANTIC MEMORY
    # ============================================================
    
    def store_semantic_memory(self, memory: SemanticMemory) -> bool:
        """Store a semantic fact"""
        if not self.connected:
            return False
        
        try:
            data = asdict(memory)
            self.db[self.SEMANTIC_MEMORY].insert_one(data)
            logger.debug(f"📚 Semantic memory stored: {memory.fact_type}")
            return True
        except PyMongoError as e:
            logger.error(f"❌ Failed to store semantic memory: {e}")
            return False
    
    def query_semantic_memory(self, fact_type: Optional[str] = None,
                             min_confidence: float = 0.0,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """Query semantic knowledge"""
        if not self.connected:
            return []
        
        try:
            query = {"confidence": {"$gte": min_confidence}}
            if fact_type:
                query["fact_type"] = fact_type
            
            facts = self.db[self.SEMANTIC_MEMORY].find(query).sort(
                "confidence", DESCENDING
            ).limit(limit)
            
            return list(facts)
        except PyMongoError as e:
            logger.error(f"❌ Failed to query semantic memory: {e}")
            return []
    
    # ============================================================
    # USER PROFILES
    # ============================================================
    
    def update_user_profile(self, profile: UserProfile) -> bool:
        """Update or create user profile"""
        if not self.connected:
            return False
        
        try:
            data = asdict(profile)
            data["last_seen"] = datetime.utcnow()
            
            self.db[self.USER_PROFILES].update_one(
                {"user_id": profile.user_id},
                {"$set": data, "$inc": {"interaction_count": 1}},
                upsert=True
            )
            
            logger.debug(f"👤 User profile updated: {profile.user_id}")
            return True
        except PyMongoError as e:
            logger.error(f"❌ Failed to update user profile: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile"""
        if not self.connected:
            return None
        
        try:
            profile = self.db[self.USER_PROFILES].find_one({"user_id": user_id})
            return profile
        except PyMongoError as e:
            logger.error(f"❌ Failed to get user profile: {e}")
            return None
    
    # ============================================================
    # PERFORMANCE METRICS
    # ============================================================
    
    def log_performance_metric(self, metric: PerformanceMetric) -> bool:
        """Log a performance metric"""
        if not self.connected:
            return False
        
        try:
            data = asdict(metric)
            self.db[self.PERFORMANCE_METRICS].insert_one(data)
            return True
        except PyMongoError as e:
            logger.error(f"❌ Failed to log performance metric: {e}")
            return False
    
    def get_performance_metrics(self, component: Optional[str] = None,
                               metric_type: Optional[str] = None,
                               hours: int = 24,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance metrics"""
        if not self.connected:
            return []
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(hours=hours)
            query = {"timestamp": {"$gte": cutoff_date}}
            
            if component:
                query["component"] = component
            if metric_type:
                query["metric_type"] = metric_type
            
            metrics = self.db[self.PERFORMANCE_METRICS].find(query).sort(
                "timestamp", DESCENDING
            ).limit(limit)
            
            return list(metrics)
        except PyMongoError as e:
            logger.error(f"❌ Failed to get performance metrics: {e}")
            return []
    
    # ============================================================
    # FEEDBACK LOGGING
    # ============================================================
    
    def log_feedback(self, session_id: str, interaction_id: str,
                    feedback_type: str, feedback_data: Dict[str, Any]) -> bool:
        """Log user feedback for learning"""
        if not self.connected:
            return False
        
        try:
            feedback_log = {
                "session_id": session_id,
                "interaction_id": interaction_id,
                "feedback_type": feedback_type,
                "feedback_data": feedback_data,
                "timestamp": datetime.utcnow()
            }
            
            self.db[self.FEEDBACK_LOGS].insert_one(feedback_log)
            logger.debug(f"📊 Feedback logged: {feedback_type}")
            return True
        except PyMongoError as e:
            logger.error(f"❌ Failed to log feedback: {e}")
            return False
    
    def get_feedback_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get feedback summary for learning"""
        if not self.connected:
            return {}
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            pipeline = [
                {"$match": {"timestamp": {"$gte": cutoff_date}}},
                {"$group": {
                    "_id": "$feedback_type",
                    "count": {"$sum": 1}
                }}
            ]
            
            results = list(self.db[self.FEEDBACK_LOGS].aggregate(pipeline))
            
            return {
                "feedback_breakdown": results,
                "period_days": days
            }
        except PyMongoError as e:
            logger.error(f"❌ Failed to get feedback summary: {e}")
            return {}
    
    # ============================================================
    # CONVERSATION HISTORY
    # ============================================================
    
    def save_conversation_turn(self, session_id: str, user_message: str,
                               robot_response: str, context: Dict[str, Any]) -> bool:
        """Save a conversation turn"""
        if not self.connected:
            return False
        
        try:
            conversation_turn = {
                "session_id": session_id,
                "user_message": user_message,
                "robot_response": robot_response,
                "context": context,
                "timestamp": datetime.utcnow()
            }
            
            self.db[self.CONVERSATION_HISTORY].insert_one(conversation_turn)
            return True
        except PyMongoError as e:
            logger.error(f"❌ Failed to save conversation turn: {e}")
            return False
    
    def get_conversation_history(self, session_id: str, 
                                limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        if not self.connected:
            return []
        
        try:
            history = self.db[self.CONVERSATION_HISTORY].find(
                {"session_id": session_id}
            ).sort("timestamp", ASCENDING).limit(limit)
            
            return list(history)
        except PyMongoError as e:
            logger.error(f"❌ Failed to get conversation history: {e}")
            return []
    
    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """Clean up old data"""
        if not self.connected:
            return {}
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            results = {}
            
            # Clean old interactions
            result = self.db[self.INTERACTIONS].delete_many(
                {"timestamp": {"$lt": cutoff_date}}
            )
            results["interactions_deleted"] = result.deleted_count
            
            # Clean old episodic memories (keep important ones)
            result = self.db[self.EPISODIC_MEMORY].delete_many({
                "timestamp": {"$lt": cutoff_date},
                "importance": {"$lt": 0.7}
            })
            results["episodic_memories_deleted"] = result.deleted_count
            
            # Clean old performance metrics
            result = self.db[self.PERFORMANCE_METRICS].delete_many(
                {"timestamp": {"$lt": cutoff_date}}
            )
            results["metrics_deleted"] = result.deleted_count
            
            logger.info(f"🧹 Cleaned up old data: {results}")
            return results
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to cleanup old data: {e}")
            return {}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.connected:
            return {}
        
        try:
            stats = {
                "interactions_count": self.db[self.INTERACTIONS].count_documents({}),
                "episodic_memories_count": self.db[self.EPISODIC_MEMORY].count_documents({}),
                "semantic_memories_count": self.db[self.SEMANTIC_MEMORY].count_documents({}),
                "user_profiles_count": self.db[self.USER_PROFILES].count_documents({}),
                "performance_metrics_count": self.db[self.PERFORMANCE_METRICS].count_documents({}),
                "feedback_logs_count": self.db[self.FEEDBACK_LOGS].count_documents({}),
                "conversation_turns_count": self.db[self.CONVERSATION_HISTORY].count_documents({}),
                "connected": self.connected
            }
            
            return stats
        except PyMongoError as e:
            logger.error(f"❌ Failed to get database stats: {e}")
            return {}
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("MongoDB connection closed")

