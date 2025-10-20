#!/usr/bin/env python3
"""
Wit.ai Intent Upload Script for Humanoid Robot Assistant

This script uploads intents and utterances to Wit.ai for training the NLP module.
Based on the successful Chapo bot implementation.

Author: Victor Ibhafidon
Date: October 2025
"""

import os
import json
import csv
import requests
import time
import logging
from typing import Dict, List, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WitAIUploader:
    """Handles uploading intents and utterances to Wit.ai"""
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        self.intent_url = "https://api.wit.ai/intents?v=20230204"
        self.utterance_url = "https://api.wit.ai/utterances?v=20230204"
        
    def create_intent(self, intent_name: str) -> bool:
        """Create a new intent in Wit.ai"""
        payload = {"name": intent_name}
        
        try:
            response = requests.post(self.intent_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                logger.info(f"✅ Created intent: {intent_name}")
                return True
            elif "already exists" in response.text:
                logger.info(f"🔁 Intent exists: {intent_name}")
                return True
            else:
                logger.error(f"❌ Failed to create intent: {intent_name} => {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Exception creating intent {intent_name}: {e}")
            return False
    
    def upload_utterance(self, utterance: str, intent: str, entities: Dict[str, Any] = None) -> bool:
        """Upload a single utterance to Wit.ai"""
        payload = {
            "text": utterance,
            "intent": intent
        }
        
        if entities:
            payload["entities"] = entities
        
        try:
            response = requests.post(self.utterance_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                logger.debug(f"✅ Uploaded utterance: {utterance[:50]}...")
                return True
            else:
                logger.error(f"❌ Failed to upload utterance: {utterance[:50]}... => {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Exception uploading utterance: {e}")
            return False
    
    def upload_batch_utterances(self, utterances: List[Dict[str, Any]], batch_size: int = 10) -> Dict[str, int]:
        """Upload multiple utterances in batches"""
        results = {"success": 0, "failed": 0}
        
        for i in range(0, len(utterances), batch_size):
            batch = utterances[i:i + batch_size]
            
            for utterance_data in batch:
                success = self.upload_utterance(
                    utterance_data["utterance"],
                    utterance_data["intent"],
                    utterance_data.get("entities")
                )
                
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
                
                # Rate limiting - Wit.ai has limits
                time.sleep(0.1)
            
            logger.info(f"📊 Batch {i//batch_size + 1}: {results['success']} success, {results['failed']} failed")
            time.sleep(1)  # Longer pause between batches
        
        return results

def load_training_data(csv_path: str) -> List[Dict[str, Any]]:
    """Load training data from CSV file"""
    training_data = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                utterance = row.get('utterance', '').strip()
                intent = row.get('intent', '').strip()
                entities_str = row.get('entities', '{}').strip()
                
                if not utterance or not intent:
                    continue
                
                # Parse entities JSON
                try:
                    entities = json.loads(entities_str) if entities_str else {}
                except json.JSONDecodeError:
                    logger.warning(f"Invalid entities JSON for utterance: {utterance[:50]}...")
                    entities = {}
                
                training_data.append({
                    "utterance": utterance,
                    "intent": intent,
                    "entities": entities
                })
        
        logger.info(f"📁 Loaded {len(training_data)} training examples from {csv_path}")
        return training_data
        
    except Exception as e:
        logger.error(f"❌ Error loading training data: {e}")
        return []

def get_unique_intents(training_data: List[Dict[str, Any]]) -> List[str]:
    """Extract unique intents from training data"""
    intents = set()
    for data in training_data:
        intents.add(data["intent"])
    return sorted(list(intents))

def main():
    """Main function to upload intents and utterances to Wit.ai"""
    
    # Configuration
    WIT_AI_ACCESS_TOKEN = os.getenv("WIT_AI_ACCESS_TOKEN")
    if not WIT_AI_ACCESS_TOKEN:
        logger.error("❌ WIT_AI_ACCESS_TOKEN environment variable not set")
        logger.info("Please set your Wit.ai access token:")
        logger.info("export WIT_AI_ACCESS_TOKEN='your_token_here'")
        return
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    csv_path = project_root / "data" / "intent_training" / "robot_intents_mega_dataset.csv"
    
    if not csv_path.exists():
        logger.error(f"❌ Training data file not found: {csv_path}")
        return
    
    # Initialize uploader
    uploader = WitAIUploader(WIT_AI_ACCESS_TOKEN)
    
    # Load training data
    logger.info("🔄 Loading training data...")
    training_data = load_training_data(str(csv_path))
    
    if not training_data:
        logger.error("❌ No training data loaded")
        return
    
    # Get unique intents
    unique_intents = get_unique_intents(training_data)
    logger.info(f"📋 Found {len(unique_intents)} unique intents")
    
    # Create intents first
    logger.info("🔄 Creating intents...")
    created_intents = 0
    for intent in unique_intents:
        if uploader.create_intent(intent):
            created_intents += 1
        time.sleep(0.5)  # Rate limiting
    
    logger.info(f"✅ Created {created_intents}/{len(unique_intents)} intents")
    
    # Upload utterances
    logger.info("🔄 Uploading utterances...")
    results = uploader.upload_batch_utterances(training_data, batch_size=5)
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 UPLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Intents created: {created_intents}/{len(unique_intents)}")
    logger.info(f"Utterances uploaded: {results['success']}")
    logger.info(f"Upload failures: {results['failed']}")
    logger.info(f"Success rate: {results['success']/(results['success']+results['failed'])*100:.1f}%")
    
    if results['failed'] == 0:
        logger.info("🎉 All data uploaded successfully!")
    else:
        logger.warning(f"⚠️  {results['failed']} uploads failed. Check logs above.")
    
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Go to https://wit.ai to train your model")
    logger.info("2. Test the trained model")
    logger.info("3. Update your NLP service configuration")

if __name__ == "__main__":
    main()
