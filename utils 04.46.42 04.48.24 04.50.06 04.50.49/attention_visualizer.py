import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
import cv2
import io
import base64

class AttentionVisualizer:
    """
    Generate attention visualizations for CNN and Vision Transformer models
    Provides clinically interpretable attention maps overlaid on chest X-rays
    """
    
    def __init__(self):
        self.colormap = 'jet'  # Heatmap colormap
        self.alpha = 0.4  # Overlay transparency
        self.figure_size = (10, 8)
        
    def generate_cnn_attention(self, image, cnn_model):
        """
        Generate CNN attention visualization using Grad-CAM style approach
        
        Args:
            image: Preprocessed image array
            cnn_model: CNN model instance
            
        Returns:
            PIL.Image: Attention visualization overlaid on original image
        """
        try:
            # Get attention features from CNN model
            attention_data = cnn_model.get_attention_features(image)
            attention_map = attention_data['attention_map']
            
            # Convert image to PIL if it's numpy array
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[-1] == 1:
                    image_array = image.squeeze(-1)
                else:
                    image_array = image
                
                # Normalize to 0-255 range
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                
                base_image = Image.fromarray(image_array).convert('RGB')
            else:
                base_image = image.convert('RGB')
            
            # Resize to standard size
            base_image = base_image.resize((224, 224))
            
            # Create attention heatmap
            attention_visualization = self._create_attention_overlay(
                base_image, attention_map, 'CNN Grad-CAM Attention'
            )
            
            return attention_visualization
            
        except Exception as e:
            return self._create_error_image(f"CNN attention generation failed: {str(e)}")
    
    def generate_vit_attention(self, image, vit_model):
        """
        Generate Vision Transformer attention visualization
        
        Args:
            image: Preprocessed image array
            vit_model: Vision Transformer model instance
            
        Returns:
            PIL.Image: Multi-head attention visualization
        """
        try:
            # Get attention maps from ViT model
            attention_data = vit_model.get_attention_maps(image)
            overall_attention = attention_data['overall_attention']
            head_importance = attention_data['head_importance']
            
            # Convert image to PIL if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[-1] == 1:
                    image_array = image.squeeze(-1)
                else:
                    image_array = image
                
                # Normalize to 0-255 range
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                
                base_image = Image.fromarray(image_array).convert('RGB')
            else:
                base_image = image.convert('RGB')
            
            # Resize to standard size
            base_image = base_image.resize((224, 224))
            
            # Create multi-head attention visualization
            attention_visualization = self._create_vit_attention_overlay(
                base_image, overall_attention, head_importance
            )
            
            return attention_visualization
            
        except Exception as e:
            return self._create_error_image(f"ViT attention generation failed: {str(e)}")
    
    def _create_attention_overlay(self, base_image, attention_map, title):
        """
        Create attention heatmap overlay on base image
        """
        # Normalize attention map
        attention_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Apply colormap
        heatmap = cm.get_cmap(self.colormap)(attention_normalized)
        heatmap_rgb = (heatmap[:, :, :3] * 255).astype(np.uint8)
        heatmap_image = Image.fromarray(heatmap_rgb)
        
        # Resize heatmap to match base image
        heatmap_image = heatmap_image.resize(base_image.size, Image.LANCZOS)
        
        # Create overlay
        overlay = Image.blend(base_image, heatmap_image, self.alpha)
        
        # Add title and create final visualization
        final_image = self._add_visualization_annotations(overlay, title, attention_map)
        
        return final_image
    
    def _create_vit_attention_overlay(self, base_image, attention_map, head_importance):
        """
        Create Vision Transformer specific attention visualization
        """
        # Create main attention overlay
        attention_overlay = self._create_attention_overlay(
            base_image, attention_map, "ViT Multi-Head Attention"
        )
        
        # Add ViT-specific annotations
        final_image = self._add_vit_annotations(attention_overlay, head_importance)
        
        return final_image
    
    def _add_visualization_annotations(self, image, title, attention_map):
        """
        Add title, legend, and clinical annotations to attention visualization
        """
        # Create a larger canvas for annotations
        canvas_width = image.width
        canvas_height = image.height + 100  # Extra space for annotations
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        
        # Paste the attention visualization
        canvas.paste(image, (0, 50))
        
        # Add title
        draw = ImageDraw.Draw(canvas)
        
        try:
            # Try to use a reasonable font size
            font_size = max(12, min(16, canvas_width // 20))
            # Use default font since custom fonts might not be available
            title_font = ImageFont.load_default()
        except:
            title_font = ImageFont.load_default()
        
        # Draw title
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (canvas_width - title_width) // 2
        draw.text((title_x, 10), title, fill='black', font=title_font)
        
        # Add attention statistics
        stats_text = f"Max Attention: {attention_map.max():.3f} | Mean: {attention_map.mean():.3f}"
        stats_bbox = draw.textbbox((0, 0), stats_text, font=title_font)
        stats_width = stats_bbox[2] - stats_bbox[0]
        stats_x = (canvas_width - stats_width) // 2
        draw.text((stats_x, canvas_height - 30), stats_text, fill='black', font=title_font)
        
        # Add color legend
        self._add_color_legend(draw, canvas_width - 150, 60, 140, 20)
        
        return canvas
    
    def _add_vit_annotations(self, image, head_importance):
        """
        Add Vision Transformer specific annotations
        """
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        
        # Add head importance information
        y_pos = image.height - 60
        importance_text = f"Most Important Heads: {np.argsort(head_importance)[-3:][::-1]}"
        draw.text((10, y_pos), importance_text, fill='white', font=font)
        
        return image
    
    def _add_color_legend(self, draw, x, y, width, height):
        """
        Add color legend for attention intensity
        """
        # Create gradient legend
        for i in range(width):
            intensity = i / width
            color_value = cm.get_cmap(self.colormap)(intensity)
            color_rgb = tuple(int(c * 255) for c in color_value[:3])
            
            # Draw vertical line for gradient
            draw.line([(x + i, y), (x + i, y + height)], fill=color_rgb)
        
        # Add labels
        font = ImageFont.load_default()
        draw.text((x, y + height + 5), "Low", fill='black', font=font)
        draw.text((x + width - 25, y + height + 5), "High", fill='black', font=font)
        draw.text((x + width//2 - 15, y - 15), "Attention", fill='black', font=font)
    
    def _create_error_image(self, error_message):
        """
        Create error image when attention generation fails
        """
        error_image = Image.new('RGB', (224, 274), 'lightgray')
        draw = ImageDraw.Draw(error_image)
        font = ImageFont.load_default()
        
        # Draw error message
        draw.text((10, 100), "Attention Visualization", fill='black', font=font)
        draw.text((10, 120), "Generation Failed", fill='red', font=font)
        draw.text((10, 150), error_message[:40] + "..." if len(error_message) > 40 else error_message, 
                 fill='gray', font=font)
        
        return error_image
    
    def generate_comparison_visualization(self, image, cnn_attention, vit_attention):
        """
        Generate side-by-side comparison of CNN and ViT attention
        """
        try:
            # Create side-by-side comparison
            comparison_width = cnn_attention.width + vit_attention.width + 20
            comparison_height = max(cnn_attention.height, vit_attention.height) + 60
            
            comparison = Image.new('RGB', (comparison_width, comparison_height), 'white')
            
            # Paste attention visualizations
            comparison.paste(cnn_attention, (10, 30))
            comparison.paste(vit_attention, (cnn_attention.width + 20, 30))
            
            # Add comparison title
            draw = ImageDraw.Draw(comparison)
            font = ImageFont.load_default()
            title = "CNN vs Vision Transformer Attention Comparison"
            title_bbox = draw.textbbox((0, 0), title, font=font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (comparison_width - title_width) // 2
            draw.text((title_x, 5), title, fill='black', font=font)
            
            return comparison
            
        except Exception as e:
            return self._create_error_image(f"Comparison generation failed: {str(e)}")
    
    def create_clinical_attention_report(self, cnn_attention_data, vit_attention_data):
        """
        Generate clinical interpretation of attention patterns
        """
        report = {
            'cnn_analysis': self._analyze_cnn_attention(cnn_attention_data),
            'vit_analysis': self._analyze_vit_attention(vit_attention_data),
            'comparison': self._compare_attention_patterns(cnn_attention_data, vit_attention_data),
            'clinical_interpretation': self._generate_clinical_interpretation(cnn_attention_data, vit_attention_data)
        }
        
        return report
    
    def _analyze_cnn_attention(self, attention_data):
        """Analyze CNN attention patterns"""
        attention_map = attention_data['attention_map']
        
        # Find areas of highest attention
        top_percentile = np.percentile(attention_map, 95)
        high_attention_mask = attention_map > top_percentile
        
        # Calculate statistics
        return {
            'max_attention': float(attention_map.max()),
            'mean_attention': float(attention_map.mean()),
            'attention_std': float(attention_map.std()),
            'high_attention_percentage': float(high_attention_mask.mean() * 100),
            'attention_distribution': 'focal' if attention_map.std() > 0.3 else 'diffuse'
        }
    
    def _analyze_vit_attention(self, attention_data):
        """Analyze Vision Transformer attention patterns"""
        overall_attention = attention_data['overall_attention']
        head_importance = attention_data['head_importance']
        
        return {
            'max_attention': float(overall_attention.max()),
            'mean_attention': float(overall_attention.mean()),
            'attention_std': float(overall_attention.std()),
            'head_diversity': float(np.std(head_importance)),
            'dominant_heads': int(np.sum(head_importance > head_importance.mean())),
            'attention_pattern': 'structured' if overall_attention.std() > 0.25 else 'uniform'
        }
    
    def _compare_attention_patterns(self, cnn_data, vit_data):
        """Compare CNN and ViT attention patterns"""
        cnn_map = cnn_data['attention_map']
        vit_map = vit_data['overall_attention']
        
        # Ensure same size for comparison
        if cnn_map.shape != vit_map.shape:
            vit_map = cv2.resize(vit_map, cnn_map.shape[::-1])
        
        # Calculate correlation
        correlation = np.corrcoef(cnn_map.flatten(), vit_map.flatten())[0, 1]
        
        # Calculate overlap in high-attention areas
        cnn_high = cnn_map > np.percentile(cnn_map, 90)
        vit_high = vit_map > np.percentile(vit_map, 90)
        overlap = np.logical_and(cnn_high, vit_high).mean()
        
        return {
            'attention_correlation': float(correlation),
            'high_attention_overlap': float(overlap * 100),
            'agreement_level': 'high' if correlation > 0.7 else 'moderate' if correlation > 0.4 else 'low'
        }
    
    def _generate_clinical_interpretation(self, cnn_data, vit_data):
        """Generate clinical interpretation of attention patterns"""
        comparison = self._compare_attention_patterns(cnn_data, vit_data)
        
        interpretations = []
        
        if comparison['attention_correlation'] > 0.7:
            interpretations.append("Both models show strong agreement in identifying suspicious regions")
        elif comparison['attention_correlation'] > 0.4:
            interpretations.append("Models show moderate agreement with some differences in focus areas")
        else:
            interpretations.append("Models focus on different image regions - consider multiple perspectives")
        
        if comparison['high_attention_overlap'] > 60:
            interpretations.append("High overlap in critical attention areas suggests consistent findings")
        else:
            interpretations.append("Limited overlap in high-attention areas suggests need for careful review")
        
        return {
            'summary': interpretations,
            'confidence_level': 'high' if comparison['attention_correlation'] > 0.6 else 'moderate',
            'recommendation': self._get_clinical_recommendation(comparison)
        }
    
    def _get_clinical_recommendation(self, comparison):
        """Get clinical recommendation based on attention analysis"""
        if comparison['attention_correlation'] > 0.7 and comparison['high_attention_overlap'] > 60:
            return "Strong model agreement supports diagnostic confidence"
        elif comparison['attention_correlation'] > 0.4:
            return "Moderate agreement - consider clinical correlation"
        else:
            return "Low model agreement - recommend additional review or imaging"
