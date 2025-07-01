import streamlit as st
import torch
import numpy as np
from PIL import Image
from models.hybrid_model import HybridCNNViTModel
from components.file_uploader import FileUploader

class AnalysisComponent:
    """
    Component to handle image upload, model inference, and result visualization
    """
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.file_uploader = FileUploader()
        self.model_loaded = False

    def load_model(self, model_path='cnn_pneumonia_model.pth'):
        """
        Load the trained hybrid model weights
        """
        self.model = HybridCNNViTModel(num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.model_loaded = True

    def preprocess_image(self, uploaded_file):
        """
        Preprocess uploaded image for model input
        """
        try:
            image = Image.open(uploaded_file).convert('RGB')
            image = image.resize((224, 224))
            image_np = np.array(image).astype(np.float32) / 255.0
            # Normalize with ImageNet mean and std
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = (image_np - mean) / std
            image_np = np.transpose(image_np, (2, 0, 1))  # C,H,W
            image_tensor = torch.tensor(image_np).unsqueeze(0).to(self.device)
            return image_tensor
        except Exception as e:
            st.error(f"Error preprocessing image: {str(e)}")
            return None

    def predict(self, image_tensor):
        """
        Run model inference and return prediction and confidence
        """
        if not self.model_loaded:
            st.warning("Model not loaded. Loading now...")
            self.load_model()

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probabilities, dim=1)
            class_names = ['Normal', 'Pneumonia']
            prediction = class_names[pred_class.item()]
            confidence_score = confidence.item()
            return prediction, confidence_score

    def render(self):
        """
        Render the analysis component UI
        """
        st.header("🩻 Pneumonia Detection Analysis")
        uploaded_file = self.file_uploader.render()

        if uploaded_file is not None:
            image_tensor = self.preprocess_image(uploaded_file)
            if image_tensor is not None:
                with st.spinner("Running model inference..."):
                    prediction, confidence = self.predict(image_tensor)
                st.markdown(f"### Prediction: **{prediction}**")
                st.markdown(f"### Confidence: **{confidence*100:.2f}%**")

                # TODO: Add attention map visualization here

        else:
            st.info("Please upload a chest X-ray image to start analysis.")
