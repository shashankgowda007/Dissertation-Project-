# Pneumonia Detection - Clinical Decision Support Platform

## Overview

This is a Streamlit-based clinical decision support web application that compares Vision Transformers (ViT) and Convolutional Neural Networks (CNNs) for pneumonia detection in chest X-ray images. The platform provides real-time analysis with attention visualization and clinical performance benchmarks aligned with healthcare standards.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with responsive layout
- **Components**: Modular component architecture with specialized UI elements
  - File uploader for medical images (DICOM, PNG, JPEG support)
  - Model comparison dashboard with interactive visualizations
  - Attention visualization components for clinical interpretability
- **Configuration**: Custom theming and server configuration for clinical environment deployment

### Backend Architecture
- **Model Layer**: Dual-model architecture comparing CNN (ResNet-50) vs Vision Transformer approaches
- **Processing Pipeline**: Specialized medical image preprocessing with DICOM support
- **Metrics Engine**: Comprehensive performance analysis against clinical benchmarks
- **Visualization Engine**: Attention map generation for model interpretability

### Key Design Decisions
- **Modular Structure**: Separated models, utilities, and components for maintainability
- **Simulated Models**: Current implementation uses simulated models for demonstration purposes
- **Clinical Focus**: Performance targets exceed radiologist baselines (94% sensitivity vs 89%)
- **Real-time Processing**: Target processing time ≤15 seconds for clinical workflow integration

## Key Components

### Models (`/models/`)
- **CNN Model**: ResNet-50 based architecture with medical domain transfer learning
- **Vision Transformer**: ViT-Base with adaptive patch size and multi-head attention
- Both models target clinical performance benchmarks with sensitivity ≥94% and specificity ≥91%

### Utilities (`/utils/`)
- **Image Processor**: Medical image preprocessing pipeline supporting DICOM and standard formats
- **Attention Visualizer**: Generates clinically interpretable attention maps using Grad-CAM style visualization
- **Metrics Calculator**: Comprehensive performance analysis comparing models against clinical benchmarks

### Components (`/components/`)
- **File Uploader**: Specialized medical image upload with validation (50MB limit, multiple format support)
- **Model Comparison**: Interactive dashboard for side-by-side model performance analysis

## Data Flow

1. **Image Upload**: User uploads chest X-ray (DICOM/PNG/JPEG) through specialized uploader
2. **Preprocessing**: Medical image processing pipeline normalizes and prepares image
3. **Model Inference**: Parallel processing through CNN and ViT models
4. **Results Generation**: Both models generate predictions with confidence scores
5. **Visualization**: Attention maps generated for clinical interpretability
6. **Comparison**: Side-by-side analysis with clinical benchmark comparison
7. **Display**: Interactive dashboard presents results with performance metrics

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for clinical interface
- **PyTorch**: Deep learning framework for model implementation
- **PIL/OpenCV**: Image processing and manipulation
- **NumPy/Pandas**: Data processing and analysis
- **Plotly**: Interactive visualizations for clinical dashboards
- **PyDICOM**: DICOM medical image format support

### Specialized Dependencies
- **Matplotlib**: Attention visualization and heatmap generation
- **Scikit-learn**: Performance metrics and statistical analysis
- **Medical Imaging**: Support for clinical image formats and preprocessing

## Deployment Strategy

### Environment Setup
- **Platform**: Replit deployment with autoscale configuration
- **Runtime**: Python 3.11 with specialized medical imaging dependencies
- **Port Configuration**: Streamlit server on port 5000
- **Dependencies**: UV package manager with PyTorch CPU optimization

### Clinical Deployment Considerations
- **Performance Targets**: Processing time ≤15 seconds for emergency department workflow
- **Reliability**: Designed for 450+ NHS A&E departments potential deployment
- **Compliance**: Aligned with clinical performance benchmarks and regulatory requirements

### Infrastructure Requirements
- **Memory**: Medical imaging processing requires sufficient RAM for DICOM handling
- **Storage**: Support for large medical image datasets
- **Network**: Low-latency deployment for real-time clinical decision support

## Enhanced Platform Features

### Database Integration
- **PostgreSQL database** integrated for comprehensive data persistence
- **Analysis Sessions**: Track complete analysis workflows with metadata
- **Model Predictions**: Store individual CNN and ViT prediction results
- **Model Comparisons**: Record agreement analysis and consensus predictions
- **Performance Metrics**: Monitor processing times and clinical benchmarks
- **Clinical Reports**: Archive generated reports for audit compliance

### Multi-Tab Interface
- **Analysis Tab**: Core pneumonia detection workflow with dual-model comparison
- **Analytics Dashboard**: System-wide performance metrics and trend analysis
- **Research Dashboard**: Academic research insights, literature review, and methodology analysis
- **Clinical Workflow**: Interactive workflow simulation and training scenarios

### Academic & Research Features
- **Literature Review Analysis**: Research gap identification and theoretical framework
- **Methodology Documentation**: Comprehensive research design and evaluation framework
- **Performance Benchmarking**: Statistical analysis and clinical validation metrics
- **Clinical Training**: Interactive workflow simulation and training scenarios

## Changelog

- June 23, 2025: Initial clinical decision support platform setup
- June 23, 2025: PostgreSQL database integration with analytics dashboard
- June 23, 2025: Enhanced platform with 4-tab interface including research and clinical workflow tools
- June 23, 2025: Fixed database NumPy type conversion issues for proper data persistence

## User Preferences

Preferred communication style: Simple, everyday language.