# Pneumonia Detection - Clinical Decision Support Platform

This project is a Streamlit-based web application designed to assist in pneumonia detection from chest X-ray images. It compares two state-of-the-art AI models: a Convolutional Neural Network (CNN) based on ResNet-50 architecture and a Vision Transformer (ViT) model. The platform provides clinically interpretable visualizations and performance metrics to support clinical decision-making.

## Features

- Upload chest X-ray images in standard formats (JPEG, PNG) and DICOM medical format.
- Medical-specific image preprocessing including contrast enhancement and normalization.
- Run inference through both CNN and Vision Transformer models with simulated realistic processing times.
- Generate attention heatmaps for both models to highlight suspicious regions in the images.
- Side-by-side model comparison with confidence scores, processing times, and detailed performance metrics.
- Dashboards for analytics, research, and clinical workflow simulation.
- Database integration for logging analysis sessions and storing results.
- Export clinical reports and attention maps for further review.

## Project Structure

- `app.py`: Main Streamlit application orchestrating the UI and workflow.
- `models/`: Contains simulated CNN and Vision Transformer model implementations.
- `utils/`: Utility modules for image processing, attention visualization, metrics calculation, and database management.
- `components/`: UI components for file uploading, model comparison, dashboards, and clinical workflow.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Upload a chest X-ray image.
3. Analyze the image using both models.
4. View attention visualizations and performance metrics.
5. Export clinical reports and attention maps as needed.

## Requirements

- Python 3.x
- Streamlit
- numpy
- Pillow
- pydicom
- opencv-python
- plotly

run
pip install -r requirements.txt

then run
streamlit run app.py

then on your browser paste these and click enter
http://localhost:5000 


## Notes

- Results should be correlated with clinical findings and are not intended for standalone diagnostic use.

## License

This project is provided as-is for research and educational purposes.
