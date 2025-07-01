import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

class ModelComparison:
    """
    Component for displaying side-by-side model comparisons
    Provides interactive visualizations and performance analysis
    """
    
    def __init__(self):
        self.colors = {
            'cnn': '#1f77b4',  # Blue
            'vit': '#ff7f0e',  # Orange
            'agreement': '#2ca02c',  # Green
            'disagreement': '#d62728'  # Red
        }
    
    def render(self, cnn_result, vit_result):
        """
        Render the complete model comparison interface
        
        Args:
            cnn_result: CNN prediction results
            vit_result: ViT prediction results
        """
        # Main comparison metrics
        self._render_prediction_comparison(cnn_result, vit_result)
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_confidence_comparison(cnn_result, vit_result)
        
        with col2:
            self._render_performance_radar(cnn_result, vit_result)
        
        # Detailed analysis
        self._render_detailed_analysis(cnn_result, vit_result)
    
    def _render_prediction_comparison(self, cnn_result, vit_result):
        """Render prediction comparison cards"""
        st.subheader("Prediction Comparison")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # CNN prediction card
            prediction_color = "red" if cnn_result['prediction'] == 'pneumonia' else "green"
            st.markdown(f"""
            <div style="border: 2px solid {prediction_color}; border-radius: 10px; padding: 15px; text-align: center;">
                <h4>🏥 CNN Model</h4>
                <h3 style="color: {prediction_color};">{cnn_result['prediction'].upper()}</h3>
                <p><strong>Confidence:</strong> {cnn_result['confidence']:.1%}</p>
                <p><strong>Raw Score:</strong> {cnn_result['raw_score']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Agreement indicator
            agreement = cnn_result['prediction'] == vit_result['prediction']
            agreement_color = self.colors['agreement'] if agreement else self.colors['disagreement']
            agreement_text = "AGREE" if agreement else "DISAGREE"
            agreement_icon = "✅" if agreement else "⚠️"
            
            st.markdown(f"""
            <div style="border: 2px solid {agreement_color}; border-radius: 10px; padding: 15px; text-align: center;">
                <h4>{agreement_icon} Models</h4>
                <h3 style="color: {agreement_color};">{agreement_text}</h3>
                <p><strong>Conf. Diff:</strong> {abs(cnn_result['confidence'] - vit_result['confidence']):.1%}</p>
                <p><strong>Score Diff:</strong> {abs(cnn_result['raw_score'] - vit_result['raw_score']):.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # ViT prediction card
            prediction_color = "red" if vit_result['prediction'] == 'pneumonia' else "green"
            st.markdown(f"""
            <div style="border: 2px solid {prediction_color}; border-radius: 10px; padding: 15px; text-align: center;">
                <h4>🤖 Vision Transformer</h4>
                <h3 style="color: {prediction_color};">{vit_result['prediction'].upper()}</h3>
                <p><strong>Confidence:</strong> {vit_result['confidence']:.1%}</p>
                <p><strong>Raw Score:</strong> {vit_result['raw_score']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_confidence_comparison(self, cnn_result, vit_result):
        """Render confidence comparison chart"""
        st.subheader("Confidence Analysis")
        
        # Create confidence comparison chart
        fig = go.Figure()
        
        # Add confidence bars
        models = ['CNN', 'Vision Transformer']
        confidences = [cnn_result['confidence'], vit_result['confidence']]
        colors = [self.colors['cnn'], self.colors['vit']]
        
        fig.add_trace(go.Bar(
            x=models,
            y=confidences,
            marker_color=colors,
            text=[f"{c:.1%}" for c in confidences],
            textposition='auto',
            name='Confidence'
        ))
        
        # Add threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                     annotation_text="Decision Threshold (50%)")
        
        # Add target confidence line
        fig.add_hline(y=0.8, line_dash="dot", line_color="green", 
                     annotation_text="High Confidence (80%)")
        
        fig.update_layout(
            title="Model Confidence Comparison",
            yaxis_title="Confidence Score",
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_radar(self, cnn_result, vit_result):
        """Render performance radar chart"""
        st.subheader("Performance Metrics")
        
        # Extract performance metrics
        categories = ['Sensitivity', 'Specificity', 'Speed', 'Interpretability', 'Confidence']
        
        # Normalize metrics for radar chart (0-1 scale)
        cnn_values = [
            cnn_result['model_info']['sensitivity'],
            cnn_result['model_info']['specificity'],
            0.85,  # Speed score (CNN typically slower)
            0.75,  # Interpretability (good with Grad-CAM)
            cnn_result['confidence']
        ]
        
        vit_values = [
            vit_result['model_info']['sensitivity'],
            vit_result['model_info']['specificity'],
            0.90,  # Speed score (ViT typically faster for inference)
            0.85,  # Interpretability (excellent with attention)
            vit_result['confidence']
        ]
        
        fig = go.Figure()
        
        # Add CNN trace
        fig.add_trace(go.Scatterpolar(
            r=cnn_values,
            theta=categories,
            fill='toself',
            name='CNN',
            line_color=self.colors['cnn'],
            fillcolor=self.colors['cnn'],
            opacity=0.3
        ))
        
        # Add ViT trace
        fig.add_trace(go.Scatterpolar(
            r=vit_values,
            theta=categories,
            fill='toself',
            name='Vision Transformer',
            line_color=self.colors['vit'],
            fillcolor=self.colors['vit'],
            opacity=0.3
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat=".0%"
                )
            ),
            showlegend=True,
            title="Model Performance Radar",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_detailed_analysis(self, cnn_result, vit_result):
        """Render detailed analysis section"""
        st.subheader("Detailed Analysis")
        
        # Create analysis tabs
        tab1, tab2, tab3 = st.tabs(["Model Architecture", "Clinical Performance", "Agreement Analysis"])
        
        with tab1:
            self._render_architecture_comparison(cnn_result, vit_result)
        
        with tab2:
            self._render_clinical_performance(cnn_result, vit_result)
        
        with tab3:
            self._render_agreement_analysis(cnn_result, vit_result)
    
    def _render_architecture_comparison(self, cnn_result, vit_result):
        """Render architecture comparison"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏥 CNN Architecture")
            st.markdown(f"**Type:** {cnn_result['model_info']['architecture']}")
            st.markdown("**Approach:** Convolutional layers with spatial hierarchies")
            st.markdown("**Attention:** Grad-CAM based visualization")
            st.markdown("**Strengths:** Well-established for medical imaging")
            st.markdown("**Training:** Transfer learning from ImageNet + Medical data")
            
            # CNN processing info
            if 'processing_metadata' in cnn_result:
                st.markdown("**Processing Details:**")
                st.code(f"Input Shape: {cnn_result['processing_metadata']['input_shape']}")
                st.code(f"Preprocessing: {cnn_result['processing_metadata']['preprocessing']}")
        
        with col2:
            st.markdown("### 🤖 Vision Transformer Architecture")
            st.markdown(f"**Type:** {vit_result['model_info']['architecture']}")
            st.markdown(f"**Patch Size:** {vit_result['model_info']['patch_size']}x{vit_result['model_info']['patch_size']}")
            st.markdown(f"**Attention Heads:** {vit_result['model_info']['num_heads']}")
            st.markdown(f"**Layers:** {vit_result['model_info']['num_layers']}")
            st.markdown("**Strengths:** Superior attention mechanisms and global context")
            
            # ViT processing info
            if 'processing_metadata' in vit_result:
                st.markdown("**Processing Details:**")
                st.code(f"Input Shape: {vit_result['processing_metadata']['input_shape']}")
                st.code(f"Patches: {vit_result['processing_metadata']['num_patches']}")
                st.code(f"Preprocessing: {vit_result['processing_metadata']['preprocessing']}")
    
    def _render_clinical_performance(self, cnn_result, vit_result):
        """Render clinical performance comparison"""
        # Performance comparison table
        performance_data = {
            'Metric': ['Sensitivity', 'Specificity', 'Current Confidence', 'Architecture'],
            'CNN': [
                f"{cnn_result['model_info']['sensitivity']:.1%}",
                f"{cnn_result['model_info']['specificity']:.1%}",
                f"{cnn_result['confidence']:.1%}",
                cnn_result['model_info']['architecture']
            ],
            'Vision Transformer': [
                f"{vit_result['model_info']['sensitivity']:.1%}",
                f"{vit_result['model_info']['specificity']:.1%}",
                f"{vit_result['confidence']:.1%}",
                vit_result['model_info']['architecture']
            ],
            'Clinical Target': ['≥94%', '≥91%', 'N/A', 'N/A'],
            'Radiologist Baseline': ['89%', '87%', 'N/A', 'Human']
        }
        
        df = pd.DataFrame(performance_data)
        st.dataframe(df, use_container_width=True)
        
        # Performance vs targets visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Sensitivity Comparison', 'Specificity Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Sensitivity comparison
        models = ['Radiologist', 'Target', 'CNN', 'ViT']
        sensitivity_values = [0.89, 0.94, cnn_result['model_info']['sensitivity'], vit_result['model_info']['sensitivity']]
        colors_sens = ['gray', 'green', self.colors['cnn'], self.colors['vit']]
        
        fig.add_trace(
            go.Bar(x=models, y=sensitivity_values, marker_color=colors_sens, name='Sensitivity'),
            row=1, col=1
        )
        
        # Specificity comparison
        specificity_values = [0.87, 0.91, cnn_result['model_info']['specificity'], vit_result['model_info']['specificity']]
        
        fig.add_trace(
            go.Bar(x=models, y=specificity_values, marker_color=colors_sens, name='Specificity'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_yaxes(tickformat=".0%", range=[0.8, 1.0])
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_agreement_analysis(self, cnn_result, vit_result):
        """Render agreement analysis"""
        # Calculate agreement metrics
        prediction_agreement = cnn_result['prediction'] == vit_result['prediction']
        confidence_diff = abs(cnn_result['confidence'] - vit_result['confidence'])
        score_diff = abs(cnn_result['raw_score'] - vit_result['raw_score'])
        
        # Agreement summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            agreement_color = "green" if prediction_agreement else "red"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border: 2px solid {agreement_color}; border-radius: 10px;">
                <h3>Prediction Agreement</h3>
                <h2 style="color: {agreement_color};">{'✅ YES' if prediction_agreement else '❌ NO'}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            conf_color = "green" if confidence_diff < 0.1 else "orange" if confidence_diff < 0.2 else "red"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border: 2px solid {conf_color}; border-radius: 10px;">
                <h3>Confidence Difference</h3>
                <h2 style="color: {conf_color};">{confidence_diff:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            score_color = "green" if score_diff < 0.1 else "orange" if score_diff < 0.2 else "red"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border: 2px solid {score_color}; border-radius: 10px;">
                <h3>Score Difference</h3>
                <h2 style="color: {score_color};">{score_diff:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Agreement interpretation
        st.markdown("### Clinical Interpretation")
        
        if prediction_agreement and confidence_diff < 0.1:
            st.success("🎯 **Strong Agreement**: Both models show consistent predictions with similar confidence levels. High diagnostic reliability.")
        elif prediction_agreement:
            st.info("✅ **Good Agreement**: Models agree on diagnosis but show different confidence levels. Consider clinical correlation.")
        else:
            st.warning("⚠️ **Disagreement**: Models provide different diagnoses. Recommend additional evaluation or expert consultation.")
        
        # Detailed agreement metrics
        st.markdown("### Agreement Metrics Details")
        
        agreement_metrics = {
            'Metric': [
                'Prediction Agreement',
                'Confidence Difference',
                'Raw Score Difference',
                'Average Confidence',
                'Higher Confidence Model'
            ],
            'Value': [
                'Yes' if prediction_agreement else 'No',
                f"{confidence_diff:.1%}",
                f"{score_diff:.3f}",
                f"{(cnn_result['confidence'] + vit_result['confidence']) / 2:.1%}",
                'CNN' if cnn_result['confidence'] > vit_result['confidence'] else 'ViT'
            ],
            'Clinical Significance': [
                'High' if prediction_agreement else 'Critical',
                'Low' if confidence_diff < 0.1 else 'Moderate' if confidence_diff < 0.2 else 'High',
                'Low' if score_diff < 0.1 else 'Moderate' if score_diff < 0.2 else 'High',
                'High' if (cnn_result['confidence'] + vit_result['confidence']) / 2 > 0.8 else 'Moderate',
                'Informational'
            ]
        }
        
        agreement_df = pd.DataFrame(agreement_metrics)
        st.dataframe(agreement_df, use_container_width=True)
