#!/usr/bin/env python3
"""
Interactive Chat with Humanoid Robot Assistant
Talk to the NLP system from the terminal!

Author: Victor Ibhafidon
Date: October 2025
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.nlp.nlp_service import NLPService, NLPRequest
import logging

# Reduce logging noise
logging.basicConfig(level=logging.WARNING)


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print(" HUMANOID ROBOT ASSISTANT - INTERACTIVE CHAT")
    print("="*80)
    print("\nWelcome! I'm your AI robot assistant.")
    print("I can understand your requests and respond naturally.")
    print("\nCommands:")
    print("  - Type your message and press Enter")
    print("  - Type 'status' to see system status")
    print("  - Type 'quit' or 'exit' to end conversation")
    print("  - Type 'clear' to start new session")
    print("="*80 + "\n")


def print_status(service):
    """Print system status"""
    status = service.get_detailed_status()
    
    print("\n" + ""*80)
    print("SYSTEM STATUS")
    print(""*80)
    
    components = status['components']
    for component, ready in components.items():
        icon = "" if ready else ""
        print(f"  {icon} {component.replace('_', ' ').title()}")
    
    print("\nActive Tiers:")
    if 'entity_extractor_tiers' in status:
        et = status['entity_extractor_tiers']
        print(f"  Entity: Tier{'1' if et.get('tier1_bert') else '3'}")
    if 'emotion_detector_tiers' in status:
        em = status['emotion_detector_tiers']
        print(f"  Emotion: Tier{'1' if em.get('tier1_transformer') else '3'}")
    if 'llm_integrator_tiers' in status:
        llm = status['llm_integrator_tiers']
        if llm.get('tier1_openai'):
            print(f"  LLM: Tier1 (OpenAI)")
        elif llm.get('tier2_ollama'):
            print(f"  LLM: Tier2 (Ollama)")
        else:
            print(f"  LLM: Tier3 (Templates)")
    
    print(""*80 + "\n")


async def chat_loop(service):
    """Main chat loop"""
    session_id = "terminal_session"
    user_id = "terminal_user"
    turn_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n Robot: Goodbye! Have a great day! \n")
                break
            
            if user_input.lower() == 'status':
                print_status(service)
                continue
            
            if user_input.lower() == 'clear':
                turn_count = 0
                session_id = f"terminal_session_{asyncio.get_event_loop().time()}"
                print("\n Started new conversation session\n")
                continue
            
            # Process with NLP service
            request = NLPRequest(
                text=user_input,
                session_id=session_id,
                user_id=user_id,
                use_rag=False,  # Set to True if you have knowledge loaded
                use_llm=True
            )
            
            # Show thinking indicator
            print(" Robot: ", end="", flush=True)
            
            # Get response
            response = await service.process(request)
            
            # Print response
            print(response.response_text)
            
            # Show details (optional - uncomment to see)
            turn_count += 1
            if turn_count <= 3:  # Show details for first few turns
                print(f"    [Intent: {response.intent or 'unknown'}, "
                      f"Emotion: {response.emotion}, "
                      f"Entities: {len(response.entities)}, "
                      f"Latency: {response.latency_ms:.0f}ms]")
            
            # Play audio if generated (optional)
            if response.response_audio_path:
                try:
                    import os
                    if sys.platform == 'darwin':  # macOS
                        os.system(f"afplay {response.response_audio_path} 2>/dev/null &")
                except:
                    pass  # Audio playback is optional
            
        except KeyboardInterrupt:
            print("\n\n Robot: Interrupted. Goodbye! \n")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            print("   (Continuing conversation...)\n")


async def main():
    """Main entry point"""
    print_banner()
    
    print("Initializing NLP system...")
    print("(This may take a few seconds on first run...)\n")
    
    try:
        service = NLPService()
        print(" NLP Service initialized!\n")
        
        # Show status
        status = service.get_component_status()
        ready_count = sum(1 for v in status.values() if v)
        total_count = len(status)
        print(f"Components ready: {ready_count}/{total_count}")
        
        if ready_count < total_count:
            print("\nNote: Some components unavailable (using fallback tiers)")
            print("Type 'status' during chat to see details\n")
        
        print(""*80)
        print(" START CHATTING BELOW")
        print(""*80 + "\n")
        
        # Start chat
        await chat_loop(service)
        
    except Exception as e:
        print(f"\n Failed to initialize NLP service: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye! \n")
        sys.exit(0)


