import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.database import DatabaseManager

class AnalyticsDashboard:
    """
    Analytics dashboard component for displaying clinical analysis statistics
    Shows performance trends, model comparisons, and system metrics
    """
    
    def __init__(self):
        self.db = DatabaseManager()
        self.colors = {
            'cnn': '#1f77b4',
            'vit': '#ff7f0e',
            'agreement': '#2ca02c',
            'disagreement': '#d62728',
            'neutral': '#7f7f7f'
        }
    
    def render(self):
        """Render the complete analytics dashboard"""
        st.header("📊 Clinical Analysis Dashboard")
        
        if self.db:
            try:
                stats = self.db.get_performance_statistics()
                recent_analyses = self.db.get_recent_analyses(limit=20)
                
                if stats and stats[0] and stats[0] > 0:
                    # Performance overview
                    self._render_performance_overview()
                    
                    # Recent analyses
                    self._render_recent_analyses()
                    
                    # Performance trends
                    self._render_performance_trends()
                    
                    # Model comparison statistics
                    self._render_model_statistics()
                else:
                    st.info("📊 Dashboard will be available after completing some analyses.")
                    st.markdown("**Next Steps:**")
                    st.markdown("1. Go to the **Analysis** tab")
                    st.markdown("2. Upload a chest X-ray image (DICOM, PNG, or JPEG)")
                    st.markdown("3. Click 'Analyze with Both Models'")
                    st.markdown("4. Return here to view performance analytics")
                    
                    # Show sample data structure for demonstration
                    st.markdown("### Expected Analytics")
                    sample_data = {
                        'Metric': ['Total Analyses', 'Average Processing Time', 'Model Agreement Rate', 'Time Target Achievement'],
                        'Value': ['0', '0.0 seconds', '0.0%', '0.0%'],
                        'Target': ['>100', '<15 seconds', '>85%', '>95%']
                    }
                    st.table(sample_data)
                    
            except Exception as e:
                st.warning(f"Analytics temporarily unavailable: {str(e)}")
                st.info("Please try refreshing the page or complete an analysis first.")
        else:
            st.error("Analytics dashboard requires database connection")
            st.info("Database connection failed. Please check configuration.")
    
    def _render_performance_overview(self):
        """Render high-level performance metrics"""
        st.subheader("System Performance Overview")
        
        stats = self.db.get_performance_statistics()
        
        if stats and stats[0]:  # Check if we have data
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Analyses", 
                    f"{int(stats[0]) if stats[0] else 0:,}",
                    help="Total number of completed pneumonia detection analyses"
                )
            
            with col2:
                avg_time = stats[1] if stats[1] else 0
                time_delta = avg_time - 15.0 if avg_time else 0
                st.metric(
                    "Avg Processing Time", 
                    f"{avg_time:.1f}s" if avg_time else "N/A",
                    delta=f"{time_delta:+.1f}s vs target" if avg_time else None,
                    delta_color="inverse"
                )
            
            with col3:
                agreement_rate = stats[2] if stats[2] else 0
                st.metric(
                    "Model Agreement Rate", 
                    f"{agreement_rate:.1f}%" if agreement_rate else "N/A",
                    help="Percentage of cases where CNN and ViT models agree"
                )
            
            with col4:
                target_rate = stats[3] if stats[3] else 0
                st.metric(
                    "Time Target Achievement", 
                    f"{target_rate:.1f}%" if target_rate else "N/A",
                    help="Percentage of analyses completed within 15-second target"
                )
        else:
            st.info("No analysis data available yet. Complete some analyses to see statistics.")
    
    def _render_recent_analyses(self):
        """Render table of recent analyses"""
        st.subheader("Recent Analyses")
        
        recent_data = self.db.get_recent_analyses(limit=20)
        
        if recent_data:
            # Convert to DataFrame for display
            df_data = []
            for row in recent_data:
                df_data.append({
                    'Timestamp': row[1].strftime('%Y-%m-%d %H:%M:%S') if row[1] else 'N/A',
                    'Image': row[2] if row[2] else 'Unknown',
                    'Status': row[3] if row[3] else 'Unknown',
                    'Agreement': '✅ Yes' if row[4] else '❌ No' if row[4] is not None else 'N/A',
                    'Confidence Level': row[5] if row[5] else 'N/A',
                    'Processing Time': f"{row[6]:.1f}s" if row[6] else 'N/A'
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent analyses found.")
    
    def _render_performance_trends(self):
        """Render performance trends over time"""
        st.subheader("Performance Trends (Last 30 Days)")
        
        trends_data = self.db.get_model_performance_trends(days=30)
        
        if trends_data:
            # Convert to DataFrame
            df_trends = pd.DataFrame(trends_data, columns=[
                'Date', 'CNN_Confidence', 'ViT_Confidence', 'Processing_Time', 'Daily_Count'
            ])
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Model Confidence Over Time',
                    'Processing Time Trend', 
                    'Daily Analysis Volume',
                    'Confidence Comparison'
                ],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Confidence trends
            fig.add_trace(
                go.Scatter(
                    x=df_trends['Date'], 
                    y=df_trends['CNN_Confidence'],
                    name='CNN Confidence',
                    line=dict(color=self.colors['cnn']),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_trends['Date'], 
                    y=df_trends['ViT_Confidence'],
                    name='ViT Confidence',
                    line=dict(color=self.colors['vit']),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
            
            # Processing time trend
            fig.add_trace(
                go.Scatter(
                    x=df_trends['Date'], 
                    y=df_trends['Processing_Time'],
                    name='Processing Time',
                    line=dict(color=self.colors['neutral']),
                    mode='lines+markers'
                ),
                row=1, col=2
            )
            
            # Daily volume
            fig.add_trace(
                go.Bar(
                    x=df_trends['Date'], 
                    y=df_trends['Daily_Count'],
                    name='Daily Analyses',
                    marker_color=self.colors['agreement']
                ),
                row=2, col=1
            )
            
            # Confidence comparison
            fig.add_trace(
                go.Scatter(
                    x=df_trends['CNN_Confidence'], 
                    y=df_trends['ViT_Confidence'],
                    mode='markers',
                    name='CNN vs ViT',
                    marker=dict(
                        size=df_trends['Daily_Count'] * 2,
                        color=self.colors['neutral'],
                        opacity=0.6
                    )
                ),
                row=2, col=2
            )
            
            # Add target line for processing time
            fig.add_hline(y=15.0, line_dash="dash", line_color="red", 
                         annotation_text="Target: 15s", row=1, col=2)
            
            # Add diagonal line for perfect correlation
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='Perfect Agreement',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=True)
            fig.update_yaxes(title_text="Confidence", range=[0, 1], row=1, col=1)
            fig.update_yaxes(title_text="Seconds", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            fig.update_yaxes(title_text="ViT Confidence", range=[0, 1], row=2, col=2)
            fig.update_xaxes(title_text="CNN Confidence", range=[0, 1], row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for trend analysis. Complete more analyses to see trends.")
    
    def _render_model_statistics(self):
        """Render detailed model comparison statistics"""
        st.subheader("Model Performance Statistics")
        
        # Get raw data for statistics
        recent_data = self.db.get_recent_analyses(limit=100)  # More data for statistics
        
        if len(recent_data) >= 5:  # Need minimum data for meaningful statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Agreement Analysis")
                
                # Calculate agreement statistics
                total_cases = len(recent_data)
                agreement_cases = sum(1 for row in recent_data if row[4])  # row[4] is prediction_agreement
                disagreement_cases = total_cases - agreement_cases
                
                # Agreement pie chart
                fig_agreement = go.Figure(data=[
                    go.Pie(
                        labels=['Agreement', 'Disagreement'],
                        values=[agreement_cases, disagreement_cases],
                        hole=0.4,
                        marker_colors=[self.colors['agreement'], self.colors['disagreement']]
                    )
                ])
                fig_agreement.update_layout(title="Model Agreement Distribution", height=300)
                st.plotly_chart(fig_agreement, use_container_width=True)
            
            with col2:
                st.markdown("### Confidence Levels")
                
                # Confidence level distribution
                confidence_levels = [row[5] for row in recent_data if row[5]]
                if confidence_levels:
                    confidence_counts = {}
                    for level in confidence_levels:
                        confidence_counts[level] = confidence_counts.get(level, 0) + 1
                    
                    fig_confidence = go.Figure(data=[
                        go.Bar(
                            x=list(confidence_counts.keys()),
                            y=list(confidence_counts.values()),
                            marker_color=self.colors['vit']
                        )
                    ])
                    fig_confidence.update_layout(
                        title="Clinical Confidence Distribution",
                        xaxis_title="Confidence Level",
                        yaxis_title="Count",
                        height=300
                    )
                    st.plotly_chart(fig_confidence, use_container_width=True)
                else:
                    st.info("No confidence data available")
            
            # Processing time analysis
            st.markdown("### Processing Time Analysis")
            
            processing_times = [row[6] for row in recent_data if row[6]]
            if processing_times:
                fig_times = go.Figure()
                
                fig_times.add_trace(go.Histogram(
                    x=processing_times,
                    nbinsx=20,
                    name='Processing Times',
                    marker_color=self.colors['neutral'],
                    opacity=0.7
                ))
                
                # Add target line
                fig_times.add_vline(
                    x=15.0, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Target: 15s"
                )
                
                fig_times.update_layout(
                    title="Processing Time Distribution",
                    xaxis_title="Processing Time (seconds)",
                    yaxis_title="Frequency",
                    height=400
                )
                
                st.plotly_chart(fig_times, use_container_width=True)
                
                # Processing time statistics
                avg_time = sum(processing_times) / len(processing_times)
                max_time = max(processing_times)
                min_time = min(processing_times)
                target_met = sum(1 for t in processing_times if t <= 15.0)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Time", f"{avg_time:.1f}s")
                with col2:
                    st.metric("Fastest Analysis", f"{min_time:.1f}s")
                with col3:
                    st.metric("Slowest Analysis", f"{max_time:.1f}s")
                with col4:
                    st.metric("Target Achievement", f"{(target_met/len(processing_times)*100):.1f}%")
        else:
            st.info("Complete at least 5 analyses to see detailed statistics.")