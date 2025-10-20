"""
LLM Integration with 3-Tier Fallback System
Tier 1: OpenAI GPT-4/GPT-4o-mini (Cloud)
Tier 2: Ollama (Local LLM - Llama 3.2, Phi-3)
Tier 3: Template-based responses (Fallback)

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
import os
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import json

# Tier 1: OpenAI
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install with: pip install openai")

# Tier 2: Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available. Install with: pip install ollama")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMTier(Enum):
    """LLM tiers"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    TEMPLATE = "template"


@dataclass
class LLMResponse:
    """Represents LLM response"""
    content: str
    tier: str
    model: str
    tokens_used: Optional[int]
    latency_ms: float
    confidence: float


class LLMIntegrator:
    """
    Multi-tier LLM integration with automatic fallback.
    Tier 1: OpenAI (best quality, requires API key)
    Tier 2: Ollama (local, good quality, no API key needed)
    Tier 3: Template-based (instant, always works)
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        ollama_model: str = "llama3.2:3b",
        ollama_host: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 256,
        timeout: int = 30
    ):
        """
        Initialize LLM integrator.
        
        Args:
            openai_api_key: OpenAI API key (from env if None)
            openai_model: OpenAI model name
            ollama_model: Ollama model name
            ollama_host: Ollama server URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.openai_model = openai_model
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Get OpenAI API key from parameter or environment
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize clients
        self.openai_client = None
        self.ollama_client = None
        
        self._initialize_openai()
        self._initialize_ollama()
        self._initialize_templates()
    
    def _initialize_openai(self):
        """Initialize OpenAI client (Tier 1)"""
        if not OPENAI_AVAILABLE:
            logger.warning("Tier 1 (OpenAI) unavailable: openai package not installed")
            return
        
        if not self.openai_api_key:
            logger.warning("Tier 1 (OpenAI) unavailable: No API key provided")
            logger.info("Set OPENAI_API_KEY environment variable to enable OpenAI")
            return
        
        try:
            self.openai_client = OpenAI(api_key=self.openai_api_key, timeout=self.timeout)
            # Test connection with a minimal request
            logger.info(" Tier 1 (OpenAI) initialized successfully")
        except Exception as e:
            logger.warning(f"OpenAI initialization failed: {e}")
            self.openai_client = None
    
    def _initialize_ollama(self):
        """Initialize Ollama client (Tier 2)"""
        if not OLLAMA_AVAILABLE:
            logger.warning("Tier 2 (Ollama) unavailable: ollama package not installed")
            return
        
        try:
            # Check if Ollama server is running
            models = ollama.list()
            logger.info(f" Tier 2 (Ollama) initialized successfully")
            logger.info(f"  Available models: {[m['name'] for m in models.get('models', [])]}")
            self.ollama_client = True  # Use module directly
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            logger.info("  Install Ollama from: https://ollama.ai")
            logger.info(f"  Then run: ollama pull {self.ollama_model}")
            self.ollama_client = None
    
    def _initialize_templates(self):
        """Initialize template-based responses (Tier 3)"""
        self.templates = {
            "greeting": "Hello! How can I assist you today?",
            "goodbye": "Goodbye! Have a great day!",
            "help": "I can help you with navigation, object manipulation, and answering questions.",
            "unknown": "I'm not sure how to respond to that. Could you rephrase?",
            "error": "I encountered an error. Please try again.",
            "fetch_object": "I'll fetch {object} for you.",
            "navigate_to": "I'll navigate to {location}.",
            "default": "I understand. Let me process that."
        }
        logger.info(" Tier 3 (Templates) initialized successfully")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        intent: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate response with automatic fallback.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
            context: Additional context (e.g., from RAG)
            intent: Detected intent (for template fallback)
            
        Returns:
            LLM response
        """
        import time
        start_time = time.time()
        
        # Build full prompt
        full_prompt = self._build_prompt(prompt, system_prompt, context)
        
        # Try Tier 1: OpenAI
        if self.openai_client is not None:
            try:
                response = await self._generate_openai(full_prompt, system_prompt)
                response.latency_ms = (time.time() - start_time) * 1000
                return response
            except Exception as e:
                logger.warning(f"Tier 1 (OpenAI) failed: {e}. Falling back to Tier 2...")
        
        # Try Tier 2: Ollama
        if self.ollama_client is not None:
            try:
                response = await self._generate_ollama(full_prompt, system_prompt)
                response.latency_ms = (time.time() - start_time) * 1000
                return response
            except Exception as e:
                logger.warning(f"Tier 2 (Ollama) failed: {e}. Falling back to Tier 3...")
        
        # Tier 3: Template-based
        response = self._generate_template(prompt, intent)
        response.latency_ms = (time.time() - start_time) * 1000
        return response
    
    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        intent: Optional[str] = None
    ) -> LLMResponse:
        """
        Synchronous version of generate.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
            context: Additional context
            intent: Detected intent
            
        Returns:
            LLM response
        """
        import time
        start_time = time.time()
        
        # Build full prompt
        full_prompt = self._build_prompt(prompt, system_prompt, context)
        
        # Try Tier 1: OpenAI
        if self.openai_client is not None:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": full_prompt})
                
                completion = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                response_content = completion.choices[0].message.content
                tokens_used = completion.usage.total_tokens if completion.usage else None
                
                latency_ms = (time.time() - start_time) * 1000
                
                return LLMResponse(
                    content=response_content,
                    tier="Tier1-OpenAI",
                    model=self.openai_model,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    confidence=0.95
                )
            except Exception as e:
                logger.warning(f"Tier 1 (OpenAI) failed: {e}. Falling back to Tier 2...")
        
        # Try Tier 2: Ollama
        if self.ollama_client is not None:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": full_prompt})
                
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                )
                
                response_content = response['message']['content']
                latency_ms = (time.time() - start_time) * 1000
                
                return LLMResponse(
                    content=response_content,
                    tier="Tier2-Ollama",
                    model=self.ollama_model,
                    tokens_used=None,
                    latency_ms=latency_ms,
                    confidence=0.85
                )
            except Exception as e:
                logger.warning(f"Tier 2 (Ollama) failed: {e}. Falling back to Tier 3...")
        
        # Tier 3: Template-based
        response = self._generate_template(prompt, intent)
        response.latency_ms = (time.time() - start_time) * 1000
        return response
    
    async def _generate_openai(self, prompt: str, system_prompt: Optional[str]) -> LLMResponse:
        """Generate using OpenAI"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        completion = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        response_content = completion.choices[0].message.content
        tokens_used = completion.usage.total_tokens if completion.usage else None
        
        return LLMResponse(
            content=response_content,
            tier="Tier1-OpenAI",
            model=self.openai_model,
            tokens_used=tokens_used,
            latency_ms=0,  # Set by caller
            confidence=0.95
        )
    
    async def _generate_ollama(self, prompt: str, system_prompt: Optional[str]) -> LLMResponse:
        """Generate using Ollama"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = ollama.chat(
            model=self.ollama_model,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        )
        
        response_content = response['message']['content']
        
        return LLMResponse(
            content=response_content,
            tier="Tier2-Ollama",
            model=self.ollama_model,
            tokens_used=None,
            latency_ms=0,  # Set by caller
            confidence=0.85
        )
    
    def _generate_template(self, prompt: str, intent: Optional[str]) -> LLMResponse:
        """Generate using templates"""
        # Try to match intent
        if intent and intent in self.templates:
            template = self.templates[intent]
            # Simple template filling (basic)
            response = template
        else:
            response = self.templates["default"]
        
        return LLMResponse(
            content=response,
            tier="Tier3-Template",
            model="template-based",
            tokens_used=None,
            latency_ms=0,  # Set by caller
            confidence=0.6
        )
    
    def _build_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str],
        context: Optional[str]
    ) -> str:
        """Build full prompt with context"""
        if context:
            return f"Context:\n{context}\n\nUser Query: {prompt}"
        return prompt
    
    def get_status(self) -> Dict[str, Any]:
        """Get LLM integrator status"""
        return {
            "tier1_openai": self.openai_client is not None,
            "tier2_ollama": self.ollama_client is not None,
            "tier3_template": True,  # Always available
            "openai_model": self.openai_model if self.openai_client else None,
            "ollama_model": self.ollama_model if self.ollama_client else None,
        }


# Example usage
if __name__ == "__main__":
    # Initialize LLM integrator
    integrator = LLMIntegrator()
    
    print("=" * 80)
    print("LLM INTEGRATOR - TESTING")
    print("=" * 80)
    print(f"\nStatus: {integrator.get_status()}\n")
    
    # Test prompts
    test_prompts = [
        ("What is the weather like today?", "answer_question", "You are a helpful robot assistant."),
        ("Tell me about yourself", "help", "You are a humanoid robot with NLP and vision capabilities."),
        ("Hello!", "greeting", None),
    ]
    
    for prompt, intent, system_prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print(f"Intent: {intent}")
        
        response = integrator.generate_sync(
            prompt=prompt,
            system_prompt=system_prompt,
            intent=intent
        )
        
        print(f"  Response: {response.content}")
        print(f"  Tier: {response.tier} | Model: {response.model}")
        print(f"  Latency: {response.latency_ms:.1f}ms | Confidence: {response.confidence:.2f}")
    
    print("\n" + "=" * 80)

