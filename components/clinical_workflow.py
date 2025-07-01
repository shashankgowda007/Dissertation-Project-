import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

class ClinicalWorkflow:
    """
    Clinical workflow simulation and training component
    Provides guided workflows for clinical users and training scenarios
    """
    
    def __init__(self):
        self.workflow_steps = {
            'emergency': [
                "Patient Arrival & Triage",
                "Initial Clinical Assessment", 
                "Chest X-ray Ordering",
                "Image Acquisition",
                "AI Analysis Processing",
                "Results Review & Interpretation",
                "Clinical Decision Making",
                "Treatment Planning"
            ],
            'routine': [
                "Patient Check-in",
                "Clinical History Review",
                "Imaging Request",
                "Quality Assurance Check",
                "AI Processing",
                "Radiologist Review",
                "Report Generation",
                "Clinical Follow-up"
            ]
        }
    
    def render(self):
        """Render the clinical workflow interface"""
        st.header("🏥 Clinical Workflow Simulator")
        st.markdown("**Interactive clinical workflow training and simulation**")
        
        # Workflow tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Workflow Simulator",
            "Training Scenarios",
            "Performance Metrics",
            "User Guide"
        ])
        
        with tab1:
            self._render_workflow_simulator()
        
        with tab2:
            self._render_training_scenarios()
        
        with tab3:
            self._render_performance_metrics()
        
        with tab4:
            self._render_user_guide()
    
    def _render_workflow_simulator(self):
        """Render interactive workflow simulator"""
        st.subheader("🔄 Interactive Workflow Simulation")
        
        # Workflow type selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            workflow_type = st.selectbox(
                "Select Workflow Type",
                ["emergency", "routine"],
                format_func=lambda x: "Emergency Department" if x == "emergency" else "Routine Screening"
            )
            
            st.markdown("### Workflow Parameters")
            urgency = st.slider("Patient Urgency Level", 1, 5, 3)
            complexity = st.selectbox("Case Complexity", ["Simple", "Moderate", "Complex"])
            
            if st.button("Start Workflow Simulation", type="primary"):
                self._run_workflow_simulation(workflow_type, urgency, complexity)
        
        with col2:
            st.markdown("### Current Workflow Status")
            if 'workflow_active' in st.session_state and st.session_state.workflow_active:
                self._display_active_workflow()
            else:
                st.info("No active workflow. Select parameters and click 'Start Workflow Simulation'")
    
    def _run_workflow_simulation(self, workflow_type, urgency, complexity):
        """Run the workflow simulation"""
        st.session_state.workflow_active = True
        st.session_state.workflow_type = workflow_type
        st.session_state.current_step = 0
        st.session_state.workflow_start_time = time.time()
        
        steps = self.workflow_steps[workflow_type]
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, step in enumerate(steps):
            progress = (i + 1) / len(steps)
            progress_bar.progress(progress)
            status_text.text(f"Step {i+1}/{len(steps)}: {step}")
            
            # Simulate step processing time based on urgency and complexity
            base_time = 0.5
            urgency_factor = (6 - urgency) * 0.2  # Higher urgency = faster
            complexity_factor = {"Simple": 1.0, "Moderate": 1.5, "Complex": 2.0}[complexity]
            
            step_time = base_time * urgency_factor * complexity_factor
            time.sleep(step_time)
            
            # Show step completion
            st.success(f"✅ Completed: {step}")
        
        # Workflow completion
        total_time = time.time() - st.session_state.workflow_start_time
        st.success(f"🎉 Workflow completed in {total_time:.1f} seconds")
        
        # Reset workflow state
        st.session_state.workflow_active = False
    
    def _display_active_workflow(self):
        """Display active workflow status"""
        workflow_type = st.session_state.get('workflow_type', 'emergency')
        current_step = st.session_state.get('current_step', 0)
        steps = self.workflow_steps[workflow_type]
        
        # Progress visualization
        fig = go.Figure()
        
        step_numbers = list(range(1, len(steps) + 1))
        completed = [1 if i <= current_step else 0.3 for i in range(len(steps))]
        
        fig.add_trace(go.Bar(
            x=step_numbers,
            y=completed,
            text=steps,
            textposition='auto',
            marker_color=['green' if c == 1 else 'lightgray' for c in completed]
        ))
        
        fig.update_layout(
            title=f"Workflow Progress: {workflow_type.title()}",
            xaxis_title="Workflow Step",
            yaxis_title="Completion Status",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_training_scenarios(self):
        """Render training scenarios"""
        st.subheader("📚 Clinical Training Scenarios")
        
        scenario_col1, scenario_col2 = st.columns(2)
        
        with scenario_col1:
            st.markdown("### Available Scenarios")
            
            scenarios = [
                {
                    "name": "Emergency Pneumonia Detection",
                    "description": "High-acuity patient with respiratory distress",
                    "difficulty": "Advanced",
                    "duration": "10 minutes"
                },
                {
                    "name": "Routine Screening Protocol",
                    "description": "Standard pneumonia screening workflow",
                    "difficulty": "Intermediate", 
                    "duration": "15 minutes"
                },
                {
                    "name": "Ambiguous Case Resolution",
                    "description": "Cases with model disagreement",
                    "difficulty": "Expert",
                    "duration": "20 minutes"
                },
                {
                    "name": "Quality Assurance Review",
                    "description": "Image quality and technical factors",
                    "difficulty": "Beginner",
                    "duration": "8 minutes"
                }
            ]
            
            for scenario in scenarios:
                with st.expander(f"{scenario['name']} ({scenario['difficulty']})"):
                    st.markdown(f"**Description:** {scenario['description']}")
                    st.markdown(f"**Duration:** {scenario['duration']}")
                    st.markdown(f"**Difficulty:** {scenario['difficulty']}")
                    if st.button(f"Start {scenario['name']}", key=scenario['name']):
                        self._start_training_scenario(scenario)
        
        with scenario_col2:
            st.markdown("### Training Progress")
            
            # Mock training progress data
            progress_data = {
                'Scenario': [
                    'Emergency Pneumonia',
                    'Routine Screening', 
                    'Ambiguous Cases',
                    'Quality Assurance'
                ],
                'Completion Rate': [85, 92, 67, 100],
                'Best Score': [88, 95, 72, 98]
            }
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Completion Rate',
                x=progress_data['Scenario'],
                y=progress_data['Completion Rate'],
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                name='Best Score',
                x=progress_data['Scenario'],
                y=progress_data['Best Score'], 
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                title="Training Progress Overview",
                yaxis_title="Score (%)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _start_training_scenario(self, scenario):
        """Start a training scenario"""
        st.info(f"Starting training scenario: {scenario['name']}")
        st.markdown("**Learning Objectives:**")
        
        if "Emergency" in scenario['name']:
            st.markdown("""
            - Rapid triage and prioritization
            - Time-critical decision making
            - Integration of AI results with clinical assessment
            - Communication with emergency team
            """)
        elif "Routine" in scenario['name']:
            st.markdown("""
            - Standard workflow protocols
            - Quality assurance procedures
            - Documentation requirements
            - Patient communication
            """)
        elif "Ambiguous" in scenario['name']:
            st.markdown("""
            - Handling model disagreements
            - Additional imaging considerations
            - Specialist consultation protocols
            - Risk assessment and management
            """)
        elif "Quality" in scenario['name']:
            st.markdown("""
            - Image quality assessment criteria
            - Technical factor optimization
            - Artifact recognition and mitigation
            - System calibration verification
            """)
    
    def _render_performance_metrics(self):
        """Render performance metrics for clinical users"""
        st.subheader("📊 Clinical Performance Metrics")
        
        # User performance tracking
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Cases Completed", "247", "+23 this week")
            st.metric("Average Accuracy", "94.2%", "+2.1% vs last month")
        
        with metrics_col2:
            st.metric("Processing Speed", "12.3 sec", "-1.2 sec improvement")
            st.metric("Model Agreement", "89.1%", "+3.4% vs baseline")
        
        with metrics_col3:
            st.metric("Training Score", "87%", "+5% since last assessment")
            st.metric("Certification Status", "Active", "Valid until Dec 2025")
        
        # Performance trends
        st.markdown("### Performance Trends")
        
        # Mock performance data over time
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
        performance_data = {
            'Date': dates,
            'Accuracy': [85 + (i % 10) + (i // 10) * 0.5 for i in range(len(dates))],
            'Speed': [15 - (i // 10) * 0.2 + (i % 5) * 0.1 for i in range(len(dates))],
            'Confidence': [80 + (i % 8) + (i // 8) * 0.3 for i in range(len(dates))]
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=performance_data['Date'],
            y=performance_data['Accuracy'],
            mode='lines+markers',
            name='Diagnostic Accuracy (%)',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=performance_data['Date'],
            y=performance_data['Confidence'],
            mode='lines+markers', 
            name='Clinical Confidence (%)',
            line=dict(color='green'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Clinical Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Accuracy (%)",
            yaxis2=dict(
                title="Confidence (%)",
                overlaying='y',
                side='right'
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_user_guide(self):
        """Render user guide and help documentation"""
        st.subheader("📖 Clinical User Guide")
        
        guide_tab1, guide_tab2, guide_tab3 = st.tabs([
            "Getting Started",
            "Best Practices", 
            "Troubleshooting"
        ])
        
        with guide_tab1:
            st.markdown("""
            ## Getting Started with AI-Assisted Pneumonia Detection
            
            ### System Overview
            This clinical decision support platform provides real-time pneumonia detection using advanced AI models to assist your diagnostic workflow.
            
            ### Basic Workflow
            1. **Image Upload**: Use the Analysis tab to upload chest X-ray images
            2. **Processing**: The system analyzes images using both CNN and Vision Transformer models
            3. **Review Results**: Examine predictions, confidence scores, and attention maps
            4. **Clinical Integration**: Correlate AI findings with clinical assessment
            
            ### Key Features
            - **Dual Model Analysis**: CNN and Vision Transformer comparison
            - **Attention Visualization**: See which image regions influence predictions
            - **Performance Tracking**: Monitor accuracy and processing metrics
            - **Clinical Compliance**: Full audit trail and regulatory compliance
            
            ### First Steps
            1. Review this user guide thoroughly
            2. Complete basic training scenarios
            3. Practice with known cases
            4. Integrate into clinical workflow gradually
            """)
        
        with guide_tab2:
            st.markdown("""
            ## Clinical Best Practices
            
            ### Image Quality Guidelines
            - Ensure proper patient positioning (PA or AP view)
            - Verify adequate inspiration (9-10 posterior ribs visible)
            - Check for motion artifacts and proper exposure
            - Confirm patient identification markers
            
            ### AI Result Interpretation
            - **High Confidence (>90%)**: Strong indication, correlate with clinical findings
            - **Moderate Confidence (70-90%)**: Consider additional imaging or specialist review
            - **Low Confidence (<70%)**: Use clinical judgment, consider repeat imaging
            
            ### Model Agreement Analysis
            - **Strong Agreement**: Both models predict same outcome with similar confidence
            - **Weak Agreement**: Models disagree or have large confidence differences
            - **Attention Analysis**: Review highlighted regions for clinical correlation
            
            ### Decision Support Guidelines
            - AI results are for decision support only, not standalone diagnosis
            - Always correlate with patient history, symptoms, and clinical examination
            - Consider additional imaging if AI results conflict with clinical suspicion
            - Document AI assistance in clinical notes as required
            
            ### Quality Assurance
            - Regular calibration checks and system validation
            - Monitor for performance degradation or bias
            - Report technical issues promptly
            - Participate in ongoing training and competency assessment
            """)
        
        with guide_tab3:
            st.markdown("""
            ## Troubleshooting Guide
            
            ### Common Issues and Solutions
            
            **Image Upload Problems**
            - Verify file format (DICOM, PNG, JPEG supported)
            - Check file size (maximum 50MB)
            - Ensure image is not corrupted
            - Try refreshing the browser
            
            **Processing Errors**
            - Check network connectivity
            - Verify image quality and format
            - Contact IT support if persistent
            - Use backup workflow if system unavailable
            
            **Unexpected Results**
            - Review image quality and positioning
            - Check for artifacts or technical factors
            - Consider patient-specific factors
            - Correlate with clinical findings
            
            **Performance Issues**
            - Clear browser cache and cookies
            - Check system resource usage
            - Report slow processing times
            - Use alternative workstation if available
            
            ### Support Contacts
            - **Technical Support**: IT Helpdesk ext. 5555
            - **Clinical Support**: AI Coordinator ext. 6666
            - **Training Questions**: Medical Education ext. 7777
            - **Emergency Backup**: Manual workflow protocol
            
            ### System Status
            - Check system status dashboard for outages
            - Subscribe to maintenance notifications
            - Maintain backup diagnostic workflows
            - Report system issues promptly
            """)