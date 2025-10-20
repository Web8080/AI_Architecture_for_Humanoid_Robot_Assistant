"""
Multimodal Fusion Engine

PURPOSE:
    Implements multimodal model fusion combining vision, language, and audio data
    following the AI concepts from modern architectures. Enables cross-modal
    understanding and reasoning for humanoid robot applications.

PIPELINE CONTEXT:
    
    Multimodal Fusion Flow:
    Vision + Language + Audio → Embeddings → Cross-Modal Attention → Fusion → Output
           ↓              ↓         ↓              ↓              ↓        ↓
    ┌─────────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐ ┌─────────┐ ┌─────────┐
    │ CLIP ViT-L  │ │ BERT    │ │ Whisper │ │ Attention   │ │ Fusion  │ │ Task    │
    │ ResNet-50   │ │ RoBERTa │ │ Wav2Vec │ │ Mechanism   │ │ Network │ │ Output  │
    │ Object Det  │ │ T5      │ │ Audio   │ │ Cross-Modal │ │ MLP     │ │ Action  │
    │ Scene Und   │ │ GPT     │ │ Emotion │ │ Transformer │ │ CNN     │ │ Response│
    └─────────────┘ └─────────┘ └─────────┘ └─────────────┘ └─────────┘ └─────────┘

WHY MULTIMODAL FUSION MATTERS:
    Current System: Separate NLP and Vision processing
    With Fusion: Unified understanding across modalities
    
    Benefits:
    - Cross-modal reasoning (see + hear + understand)
    - Better context understanding
    - Improved task performance
    - Human-like perception
    - Robust to single-modality failures

HOW IT WORKS:
    1. Embeddings: Convert each modality to vector representations
    2. Cross-Modal Attention: Learn relationships between modalities
    3. Fusion: Combine information using attention mechanisms
    4. Task-Specific Output: Generate modality-appropriate responses

INTEGRATION WITH EXISTING SYSTEM:
    - Uses existing NLP embeddings (BERT, sentence-transformers)
    - Uses existing Vision embeddings (CLIP, object features)
    - Adds audio embeddings (Whisper, Wav2Vec)
    - Enables cross-modal tasks (VQA, audio-visual scene understanding)

RELATED FILES:
    - src/nlp/nlp_service.py: Language embeddings
    - src/vision/vision_service.py: Visual embeddings
    - src/agents/ai_agent.py: Main orchestrator
    - configs/base/system_config.yaml: Fusion configuration

USAGE:
    # Initialize multimodal fusion
    fusion = MultimodalFusion(config)
    
    # Process multi-modal input
    result = await fusion.fuse_modalities(
        vision_features=vision_embeddings,
        text_features=text_embeddings,
        audio_features=audio_embeddings
    )
    
    # Cross-modal tasks
    vqa_result = await fusion.visual_question_answering(
        image=image,
        question="What objects do you see?"
    )

Author: Victor Ibhafidon
Date: October 2025
Version: 1.0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
import cv2
from PIL import Image
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class ModalityEmbedding:
    """Embedding for a single modality"""
    modality: str  # 'vision', 'text', 'audio'
    embedding: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None


@dataclass
class FusionResult:
    """Result from multimodal fusion"""
    fused_embedding: np.ndarray
    modality_contributions: Dict[str, float]
    attention_weights: Dict[str, np.ndarray]
    confidence: float
    task_outputs: Dict[str, Any]
    timestamp: float


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for learning relationships between modalities
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-modal attention
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor [batch, seq_len, embed_dim]
            value: Value tensor [batch, seq_len, embed_dim]
        
        Returns:
            Attended features [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = query.size()
        
        # Project to Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class MultimodalFusionNetwork(nn.Module):
    """
    Neural network for multimodal fusion
    """
    
    def __init__(self, 
                 vision_dim: int = 512,
                 text_dim: int = 768,
                 audio_dim: int = 256,
                 hidden_dim: int = 512,
                 output_dim: int = 256,
                 num_attention_layers: int = 3):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Projection layers to common dimension
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Cross-modal attention layers
        self.attention_layers = nn.ModuleList([
            CrossModalAttention(hidden_dim) for _ in range(num_attention_layers)
        ])
        
        # Fusion layers
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Task-specific heads
        self.vqa_head = nn.Linear(output_dim, 1000)  # VQA vocabulary
        self.action_head = nn.Linear(output_dim, 50)  # Action classes
        self.emotion_head = nn.Linear(output_dim, 7)  # Emotion classes
    
    def forward(self, 
                vision_features: torch.Tensor,
                text_features: torch.Tensor,
                audio_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multimodal fusion network
        
        Args:
            vision_features: [batch, vision_dim]
            text_features: [batch, text_dim]
            audio_features: [batch, audio_dim]
        
        Returns:
            Dictionary with fused features and task outputs
        """
        batch_size = vision_features.size(0)
        
        # Project to common dimension
        vision_proj = self.vision_proj(vision_features)  # [batch, hidden_dim]
        text_proj = self.text_proj(text_features)        # [batch, hidden_dim]
        audio_proj = self.audio_proj(audio_features)     # [batch, hidden_dim]
        
        # Stack for attention (treat as sequence)
        modalities = torch.stack([vision_proj, text_proj, audio_proj], dim=1)  # [batch, 3, hidden_dim]
        
        # Apply cross-modal attention
        attended_features = modalities
        attention_weights = []
        
        for attention_layer in self.attention_layers:
            attended_features, attn_weights = attention_layer(
                attended_features, attended_features, attended_features
            )
            attention_weights.append(attn_weights)
        
        # Flatten for fusion
        fused_input = attended_features.view(batch_size, -1)  # [batch, hidden_dim * 3]
        
        # Fusion MLP
        fused_features = self.fusion_mlp(fused_input)  # [batch, output_dim]
        
        # Task-specific outputs
        vqa_output = self.vqa_head(fused_features)
        action_output = self.action_head(fused_features)
        emotion_output = self.emotion_head(fused_features)
        
        return {
            'fused_features': fused_features,
            'attention_weights': attention_weights,
            'vqa_output': vqa_output,
            'action_output': action_output,
            'emotion_output': emotion_output
        }


class MultimodalFusion:
    """
    Multimodal fusion engine for combining vision, language, and audio
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multimodal fusion engine
        
        Args:
            config: Configuration for fusion models and parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize embedding models
        self._init_embedding_models()
        
        # Initialize fusion network
        self.fusion_network = MultimodalFusionNetwork(
            vision_dim=config.get('vision_dim', 512),
            text_dim=config.get('text_dim', 768),
            audio_dim=config.get('audio_dim', 256),
            hidden_dim=config.get('hidden_dim', 512),
            output_dim=config.get('output_dim', 256),
            num_attention_layers=config.get('num_attention_layers', 3)
        ).to(self.device)
        
        # Load pre-trained weights if available
        self._load_pretrained_weights()
        
        logger.info(f"Multimodal fusion initialized on {self.device}")
    
    def _init_embedding_models(self):
        """Initialize embedding models for each modality"""
        try:
            # Vision embeddings (CLIP)
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            logger.info("CLIP model loaded for vision embeddings")
        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
        
        try:
            # Text embeddings (BERT)
            self.text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.text_model = AutoModel.from_pretrained("bert-base-uncased")
            self.text_model.to(self.device)
            logger.info("BERT model loaded for text embeddings")
        except Exception as e:
            logger.warning(f"Failed to load BERT model: {e}")
            self.text_tokenizer = None
            self.text_model = None
        
        # Audio embeddings (placeholder - would use Whisper/Wav2Vec in production)
        self.audio_model = None
        logger.info("Audio embedding model placeholder initialized")
    
    def _load_pretrained_weights(self):
        """Load pre-trained fusion network weights if available"""
        weights_path = self.config.get('fusion_weights_path')
        if weights_path and Path(weights_path).exists():
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.fusion_network.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded pre-trained fusion weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained weights: {e}")
        else:
            logger.info("No pre-trained weights found, using random initialization")
    
    async def extract_vision_embedding(self, image: Union[np.ndarray, Image.Image, str]) -> ModalityEmbedding:
        """
        Extract vision embedding from image
        
        Args:
            image: Input image (numpy array, PIL Image, or path)
        
        Returns:
            Vision embedding
        """
        try:
            if self.clip_model is None:
                # Fallback: simple feature extraction
                if isinstance(image, str):
                    image = cv2.imread(image)
                if isinstance(image, np.ndarray):
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                
                # Simple feature extraction (placeholder)
                features = np.random.randn(512).astype(np.float32)
                return ModalityEmbedding(
                    modality='vision',
                    embedding=features,
                    confidence=0.5,
                    metadata={'method': 'fallback'}
                )
            
            # Use CLIP for vision embedding
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Process with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                vision_features = self.clip_model.get_image_features(**inputs)
                vision_features = F.normalize(vision_features, p=2, dim=1)
            
            return ModalityEmbedding(
                modality='vision',
                embedding=vision_features.cpu().numpy().flatten(),
                confidence=0.9,
                metadata={'method': 'clip', 'model': 'vit-base-patch32'}
            )
            
        except Exception as e:
            logger.error(f"Vision embedding extraction failed: {e}")
            # Return zero embedding as fallback
            return ModalityEmbedding(
                modality='vision',
                embedding=np.zeros(512, dtype=np.float32),
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def extract_text_embedding(self, text: str) -> ModalityEmbedding:
        """
        Extract text embedding from text
        
        Args:
            text: Input text
        
        Returns:
            Text embedding
        """
        try:
            if self.text_model is None:
                # Fallback: simple text features
                features = np.random.randn(768).astype(np.float32)
                return ModalityEmbedding(
                    modality='text',
                    embedding=features,
                    confidence=0.5,
                    metadata={'method': 'fallback'}
                )
            
            # Use BERT for text embedding
            inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                # Use [CLS] token embedding
                text_features = outputs.last_hidden_state[:, 0, :]
                text_features = F.normalize(text_features, p=2, dim=1)
            
            return ModalityEmbedding(
                modality='text',
                embedding=text_features.cpu().numpy().flatten(),
                confidence=0.9,
                metadata={'method': 'bert', 'model': 'base-uncased'}
            )
            
        except Exception as e:
            logger.error(f"Text embedding extraction failed: {e}")
            return ModalityEmbedding(
                modality='text',
                embedding=np.zeros(768, dtype=np.float32),
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def extract_audio_embedding(self, audio: Union[np.ndarray, str]) -> ModalityEmbedding:
        """
        Extract audio embedding from audio
        
        Args:
            audio: Input audio (numpy array or path)
        
        Returns:
            Audio embedding
        """
        try:
            # Placeholder for audio embedding (would use Whisper/Wav2Vec in production)
            if isinstance(audio, str):
                # Load audio file (placeholder)
                features = np.random.randn(256).astype(np.float32)
            else:
                # Process audio array (placeholder)
                features = np.random.randn(256).astype(np.float32)
            
            return ModalityEmbedding(
                modality='audio',
                embedding=features,
                confidence=0.7,
                metadata={'method': 'placeholder', 'note': 'Would use Whisper/Wav2Vec in production'}
            )
            
        except Exception as e:
            logger.error(f"Audio embedding extraction failed: {e}")
            return ModalityEmbedding(
                modality='audio',
                embedding=np.zeros(256, dtype=np.float32),
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def fuse_modalities(self, 
                            vision_features: Optional[ModalityEmbedding] = None,
                            text_features: Optional[ModalityEmbedding] = None,
                            audio_features: Optional[ModalityEmbedding] = None) -> FusionResult:
        """
        Fuse multiple modality embeddings
        
        Args:
            vision_features: Vision embedding
            text_features: Text embedding
            audio_features: Audio embedding
        
        Returns:
            Fused result
        """
        try:
            # Prepare features for fusion
            available_modalities = []
            modality_embeddings = []
            
            if vision_features is not None:
                available_modalities.append('vision')
                modality_embeddings.append(vision_features.embedding)
            
            if text_features is not None:
                available_modalities.append('text')
                modality_embeddings.append(text_features.embedding)
            
            if audio_features is not None:
                available_modalities.append('audio')
                modality_embeddings.append(audio_features.embedding)
            
            if not modality_embeddings:
                raise ValueError("No modality features provided")
            
            # Convert to tensors
            vision_tensor = torch.tensor(vision_features.embedding if vision_features else np.zeros(512)).unsqueeze(0).to(self.device)
            text_tensor = torch.tensor(text_features.embedding if text_features else np.zeros(768)).unsqueeze(0).to(self.device)
            audio_tensor = torch.tensor(audio_features.embedding if audio_features else np.zeros(256)).unsqueeze(0).to(self.device)
            
            # Forward pass through fusion network
            with torch.no_grad():
                fusion_output = self.fusion_network(vision_tensor, text_tensor, audio_tensor)
            
            # Extract results
            fused_embedding = fusion_output['fused_features'].cpu().numpy().flatten()
            attention_weights = fusion_output['attention_weights']
            
            # Calculate modality contributions
            modality_contributions = {}
            if vision_features:
                modality_contributions['vision'] = vision_features.confidence
            if text_features:
                modality_contributions['text'] = text_features.confidence
            if audio_features:
                modality_contributions['audio'] = audio_features.confidence
            
            # Normalize contributions
            total_confidence = sum(modality_contributions.values())
            if total_confidence > 0:
                modality_contributions = {k: v/total_confidence for k, v in modality_contributions.items()}
            
            # Task outputs
            task_outputs = {
                'vqa_logits': fusion_output['vqa_output'].cpu().numpy(),
                'action_logits': fusion_output['action_output'].cpu().numpy(),
                'emotion_logits': fusion_output['emotion_output'].cpu().numpy()
            }
            
            # Overall confidence
            overall_confidence = sum(modality_contributions.values()) / len(modality_contributions)
            
            return FusionResult(
                fused_embedding=fused_embedding,
                modality_contributions=modality_contributions,
                attention_weights={'attention': attention_weights},
                confidence=overall_confidence,
                task_outputs=task_outputs,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Modality fusion failed: {e}")
            # Return fallback result
            return FusionResult(
                fused_embedding=np.zeros(256, dtype=np.float32),
                modality_contributions={},
                attention_weights={},
                confidence=0.0,
                task_outputs={},
                timestamp=time.time()
            )
    
    async def visual_question_answering(self, 
                                      image: Union[np.ndarray, Image.Image, str],
                                      question: str) -> Dict[str, Any]:
        """
        Visual Question Answering using multimodal fusion
        
        Args:
            image: Input image
            question: Question about the image
        
        Returns:
            VQA result
        """
        try:
            # Extract embeddings
            vision_emb = await self.extract_vision_embedding(image)
            text_emb = await self.extract_text_embedding(question)
            
            # Fuse modalities
            fusion_result = await self.fuse_modalities(
                vision_features=vision_emb,
                text_features=text_emb
            )
            
            # Get VQA prediction
            vqa_logits = fusion_result.task_outputs.get('vqa_logits', np.zeros(1000))
            predicted_class = np.argmax(vqa_logits)
            
            # Simple answer mapping (would use proper VQA vocabulary in production)
            answer_mapping = {
                0: "yes", 1: "no", 2: "cup", 3: "bottle", 4: "phone", 5: "person",
                6: "table", 7: "chair", 8: "book", 9: "remote", 10: "keys"
            }
            
            answer = answer_mapping.get(predicted_class, "unknown")
            confidence = float(np.max(F.softmax(torch.tensor(vqa_logits), dim=0)))
            
            return {
                'answer': answer,
                'confidence': confidence,
                'question': question,
                'fusion_confidence': fusion_result.confidence,
                'modality_contributions': fusion_result.modality_contributions
            }
            
        except Exception as e:
            logger.error(f"VQA failed: {e}")
            return {
                'answer': 'error',
                'confidence': 0.0,
                'question': question,
                'error': str(e)
            }
    
    async def cross_modal_retrieval(self, 
                                  query_modality: str,
                                  query_data: Any,
                                  target_modality: str,
                                  candidate_data: List[Any]) -> List[Dict[str, Any]]:
        """
        Cross-modal retrieval (e.g., text-to-image, image-to-text)
        
        Args:
            query_modality: 'text', 'vision', or 'audio'
            query_data: Query data
            target_modality: Target modality to retrieve
            candidate_data: List of candidate data
        
        Returns:
            Ranked retrieval results
        """
        try:
            # Extract query embedding
            if query_modality == 'text':
                query_emb = await self.extract_text_embedding(query_data)
            elif query_modality == 'vision':
                query_emb = await self.extract_vision_embedding(query_data)
            elif query_modality == 'audio':
                query_emb = await self.extract_audio_embedding(query_data)
            else:
                raise ValueError(f"Unsupported query modality: {query_modality}")
            
            # Extract candidate embeddings
            candidate_embeddings = []
            for candidate in candidate_data:
                if target_modality == 'text':
                    emb = await self.extract_text_embedding(candidate)
                elif target_modality == 'vision':
                    emb = await self.extract_vision_embedding(candidate)
                elif target_modality == 'audio':
                    emb = await self.extract_audio_embedding(candidate)
                else:
                    raise ValueError(f"Unsupported target modality: {target_modality}")
                
                candidate_embeddings.append(emb)
            
            # Compute similarities
            results = []
            query_vec = query_emb.embedding
            
            for i, candidate_emb in enumerate(candidate_embeddings):
                candidate_vec = candidate_emb.embedding
                
                # Cosine similarity
                similarity = np.dot(query_vec, candidate_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(candidate_vec)
                )
                
                results.append({
                    'index': i,
                    'data': candidate_data[i],
                    'similarity': float(similarity),
                    'confidence': candidate_emb.confidence
                })
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Cross-modal retrieval failed: {e}")
            return []
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get fusion engine statistics"""
        return {
            'device': str(self.device),
            'vision_model_loaded': self.clip_model is not None,
            'text_model_loaded': self.text_model is not None,
            'audio_model_loaded': self.audio_model is not None,
            'fusion_network_parameters': sum(p.numel() for p in self.fusion_network.parameters()),
            'timestamp': time.time()
        }


# Example usage and testing
async def main():
    """Example usage of multimodal fusion"""
    config = {
        'vision_dim': 512,
        'text_dim': 768,
        'audio_dim': 256,
        'hidden_dim': 512,
        'output_dim': 256,
        'num_attention_layers': 3
    }
    
    # Initialize fusion engine
    fusion = MultimodalFusion(config)
    
    # Test vision embedding
    vision_emb = await fusion.extract_vision_embedding(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    print(f"Vision embedding shape: {vision_emb.embedding.shape}")
    
    # Test text embedding
    text_emb = await fusion.extract_text_embedding("What objects do you see in this image?")
    print(f"Text embedding shape: {text_emb.embedding.shape}")
    
    # Test audio embedding
    audio_emb = await fusion.extract_audio_embedding(np.random.randn(16000))
    print(f"Audio embedding shape: {audio_emb.embedding.shape}")
    
    # Test fusion
    fusion_result = await fusion.fuse_modalities(
        vision_features=vision_emb,
        text_features=text_emb,
        audio_features=audio_emb
    )
    print(f"Fused embedding shape: {fusion_result.fused_embedding.shape}")
    print(f"Modality contributions: {fusion_result.modality_contributions}")
    
    # Test VQA
    vqa_result = await fusion.visual_question_answering(
        image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        question="What do you see?"
    )
    print(f"VQA result: {vqa_result}")
    
    # Get statistics
    stats = fusion.get_fusion_statistics()
    print(f"Fusion statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
