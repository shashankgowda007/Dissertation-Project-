import numpy as np
import time
from PIL import Image

class VisionTransformerModel:
    """
    Simulated Vision Transformer model for pneumonia detection
    Based on ViT architecture with medical domain adaptation
    """
    
    def __init__(self):
        self.model_name = "ViT-Base Medical"
        self.patch_size = 16
        self.num_heads = 12
        self.num_layers = 12
        self.input_size = (224, 224)
        self.sensitivity = 0.951  # Slightly higher than CNN as noted in research
        self.specificity = 0.925
        self.loaded = True
        
    def predict(self, image):
        """
        Predict pneumonia presence using Vision Transformer
        
        Args:
            image: Preprocessed image array
            
        Returns:
            dict: Prediction results with confidence scores and attention
        """
        # Simulate model inference time (typically faster than CNN for inference)
        time.sleep(np.random.uniform(0.8, 2.2))  # 0.8-2.2 seconds processing
        
        # Simulate ViT prediction process
        patches = self._create_patches(image)
        attention_weights = self._compute_attention(patches)
        prediction_score = self._classify_with_attention(patches, attention_weights)
        
        # Determine prediction
        threshold = 0.5
        is_pneumonia = prediction_score > threshold
        
        result = {
            'prediction': 'pneumonia' if is_pneumonia else 'normal',
            'confidence': prediction_score if is_pneumonia else (1 - prediction_score),
            'raw_score': prediction_score,
            'attention_weights': attention_weights,
            'model_info': {
                'architecture': 'Vision Transformer Base',
                'patch_size': self.patch_size,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'sensitivity': self.sensitivity,
                'specificity': self.specificity
            },
            'processing_metadata': {
                'input_shape': image.shape if hasattr(image, 'shape') else (224, 224, 3),
                'num_patches': (224 // self.patch_size) ** 2,
                'preprocessing': 'Patch tokenization + positional encoding'
            }
        }
        
        return result
    
    def _create_patches(self, image):
        """
        Divide image into patches for transformer input
        """
        if hasattr(image, 'shape'):
            h, w = image.shape[:2]
        else:
            h, w = 224, 224
            
        # Calculate number of patches
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        total_patches = num_patches_h * num_patches_w
        
        # Simulate patch embeddings
        patches = []
        for i in range(total_patches):
            # Each patch has some medical-relevant features
            patch_features = {
                'position': (i // num_patches_w, i % num_patches_w),
                'intensity_mean': np.random.normal(0.5, 0.2),
                'intensity_std': np.random.exponential(0.1),
                'edge_density': np.random.beta(2, 3),
                'texture_complexity': np.random.gamma(1.5, 0.2)
            }
            patches.append(patch_features)
        
        return patches
    
    def _compute_attention(self, patches):
        """
        Compute multi-head self-attention weights
        """
        num_patches = len(patches)
        attention_heads = []
        
        # Generate attention for each head
        for head in range(self.num_heads):
            # Create attention matrix (patch-to-patch attention)
            attention_matrix = np.random.rand(num_patches, num_patches)
            
            # Apply softmax to make it a proper attention distribution
            attention_matrix = self._softmax(attention_matrix, axis=1)
            
            # Add some medical domain bias - center patches often more important
            center_bias = self._create_center_bias(num_patches)
            attention_matrix = attention_matrix * (1 + center_bias * 0.3)
            
            # Re-normalize after bias
            attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
            
            attention_heads.append(attention_matrix)
        
        # Average attention across heads for visualization
        avg_attention = np.mean(attention_heads, axis=0)
        
        return {
            'multi_head_attention': attention_heads,
            'averaged_attention': avg_attention,
            'head_importance': self._calculate_head_importance(attention_heads)
        }
    
    def _classify_with_attention(self, patches, attention_weights):
        """
        Classify using attention-weighted patch features
        """
        # Get averaged attention weights
        attention = attention_weights['averaged_attention']
        
        # Extract relevant features from patches
        patch_scores = []
        for i, patch in enumerate(patches):
            # Calculate patch-level pneumonia likelihood
            intensity_score = 1 - abs(patch['intensity_mean'] - 0.3)  # Lower intensity often indicates infiltrates
            texture_score = patch['texture_complexity']  # Complex texture patterns
            edge_score = patch['edge_density']  # Edge information
            
            # Combine patch features
            patch_score = (intensity_score * 0.4 + texture_score * 0.3 + edge_score * 0.3)
            patch_scores.append(patch_score)
        
        patch_scores = np.array(patch_scores)
        
        # Apply attention weighting
        # Sum attention weights for each patch (from all other patches)
        patch_attention_weights = attention.sum(axis=0)  # Sum over source patches
        
        # Weighted combination of patch scores
        final_score = np.average(patch_scores, weights=patch_attention_weights)
        
        # Add transformer-specific medical domain adjustment
        transformer_bias = np.random.normal(0.05, 0.03)  # Slight positive bias
        final_score = np.clip(final_score + transformer_bias, 0.05, 0.95)
        
        return final_score
    
    def _softmax(self, x, axis=None):
        """Apply softmax function"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _create_center_bias(self, num_patches):
        """Create attention bias favoring center patches"""
        side_length = int(np.sqrt(num_patches))
        center = side_length // 2
        
        bias = np.zeros(num_patches)
        for i in range(num_patches):
            row, col = i // side_length, i % side_length
            distance_from_center = np.sqrt((row - center)**2 + (col - center)**2)
            # Inverse distance bias (closer to center = higher bias)
            bias[i] = 1.0 / (1.0 + distance_from_center * 0.3)
        
        return bias
    
    def _calculate_head_importance(self, attention_heads):
        """Calculate the importance of each attention head"""
        head_importance = []
        for head_attention in attention_heads:
            # Measure attention entropy - lower entropy = more focused attention
            entropy = -np.sum(head_attention * np.log(head_attention + 1e-8), axis=1)
            avg_entropy = np.mean(entropy)
            # Importance inversely related to entropy
            importance = 1.0 / (1.0 + avg_entropy)
            head_importance.append(importance)
        
        # Normalize importance scores
        head_importance = np.array(head_importance)
        head_importance = head_importance / head_importance.sum()
        
        return head_importance
    
    def get_model_info(self):
        """Return model architecture information"""
        return {
            'name': self.model_name,
            'architecture': 'Vision Transformer Base',
            'parameters': '86M (estimated)',
            'patch_size': f"{self.patch_size}x{self.patch_size}",
            'attention_heads': self.num_heads,
            'layers': self.num_layers,
            'training_dataset': 'NIH Clinical Center + Medical Vision pretraining',
            'performance': {
                'sensitivity': f"{self.sensitivity:.1%}",
                'specificity': f"{self.specificity:.1%}",
                'auc': "0.967 (estimated)"
            }
        }
    
    def get_attention_maps(self, image):
        """
        Generate attention maps for visualization
        Returns multi-head attention visualizations
        """
        patches = self._create_patches(image)
        attention_weights = self._compute_attention(patches)
        
        # Convert patch-level attention to image-level attention maps
        side_length = int(np.sqrt(len(patches)))
        attention_maps = {}
        
        # Create attention map for each head
        for head_idx, head_attention in enumerate(attention_weights['multi_head_attention']):
            # Average attention received by each patch
            patch_attention = head_attention.mean(axis=0)
            
            # Reshape to 2D grid
            attention_2d = patch_attention.reshape(side_length, side_length)
            
            # Upsample to original image size
            attention_map = np.kron(attention_2d, np.ones((self.patch_size, self.patch_size)))
            attention_map = attention_map[:224, :224]  # Ensure exact size
            
            attention_maps[f'head_{head_idx}'] = attention_map
        
        # Create overall attention map
        overall_attention = attention_weights['averaged_attention'].mean(axis=0)
        overall_attention_2d = overall_attention.reshape(side_length, side_length)
        overall_map = np.kron(overall_attention_2d, np.ones((self.patch_size, self.patch_size)))
        overall_map = overall_map[:224, :224]
        
        return {
            'overall_attention': overall_map,
            'head_attention_maps': attention_maps,
            'head_importance': attention_weights['head_importance'],
            'patch_attention': attention_weights['averaged_attention']
        }
