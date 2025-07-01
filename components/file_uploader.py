import streamlit as st
from PIL import Image
import io

class FileUploader:
    """
    Specialized file uploader component for medical images
    Handles DICOM and standard image formats with validation
    """
    
    def __init__(self):
        self.supported_formats = ['png', 'jpg', 'jpeg', 'dcm', 'dicom']
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit
        
    def render(self):
        """
        Render the file upload interface
        
        Returns:
            uploaded file object or None
        """
        st.markdown("### Upload Chest X-ray Image")
        st.markdown("**Supported formats:** DICOM (.dcm), PNG, JPEG")
        st.markdown("**Maximum file size:** 50MB")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=self.supported_formats,
            help="Upload a chest X-ray in DICOM, PNG, or JPEG format"
        )
        
        if uploaded_file is not None:
            # File validation
            validation_result = self._validate_file(uploaded_file)
            
            if validation_result['is_valid']:
                self._display_file_info(uploaded_file)
                
                # Show preview if it's a standard image format
                if uploaded_file.type.startswith('image/'):
                    try:
                        image = Image.open(uploaded_file)
                        st.markdown("#### Preview")
                        st.image(image, caption="Uploaded image preview", width=300)
                        uploaded_file.seek(0)  # Reset file pointer
                    except Exception as e:
                        st.warning(f"Could not display preview: {str(e)}")
                
                return uploaded_file
            else:
                # Display validation errors
                for error in validation_result['errors']:
                    st.error(f"❌ {error}")
                
                for warning in validation_result['warnings']:
                    st.warning(f"⚠️ {warning}")
                
                return None
        
        else:
            # Display upload instructions
            self._display_upload_instructions()
            return None
    
    def _validate_file(self, uploaded_file):
        """
        Validate uploaded file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            dict: Validation results with errors and warnings
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check file size
        if uploaded_file.size > self.max_file_size:
            validation['errors'].append(f"File size ({uploaded_file.size / 1024 / 1024:.1f}MB) exceeds maximum limit (50MB)")
            validation['is_valid'] = False
        
        # Check file format
        file_extension = uploaded_file.name.lower().split('.')[-1]
        if file_extension not in self.supported_formats:
            validation['errors'].append(f"Unsupported file format: .{file_extension}")
            validation['is_valid'] = False
        
        # Additional validation for image files
        if uploaded_file.type.startswith('image/'):
            try:
                # Try to open image to validate it's not corrupted
                image = Image.open(uploaded_file)
                width, height = image.size
                
                # Check minimum dimensions
                if width < 64 or height < 64:
                    validation['warnings'].append("Image resolution is very low, may affect accuracy")
                
                # Check if it's grayscale (typical for chest X-rays)
                if image.mode not in ['L', 'RGB', 'RGBA']:
                    validation['warnings'].append("Unusual image mode detected")
                
                # Reset file pointer
                uploaded_file.seek(0)
                
            except Exception as e:
                validation['errors'].append(f"Invalid or corrupted image file: {str(e)}")
                validation['is_valid'] = False
        
        # DICOM file validation
        elif file_extension in ['dcm', 'dicom']:
            try:
                # Basic DICOM validation - check for DICOM header
                uploaded_file.seek(128)  # DICOM header starts at byte 128
                dicom_header = uploaded_file.read(4)
                uploaded_file.seek(0)  # Reset file pointer
                
                if dicom_header != b'DICM':
                    validation['warnings'].append("File may not be a valid DICOM file")
                
            except Exception as e:
                validation['warnings'].append(f"Could not validate DICOM file: {str(e)}")
        
        return validation
    
    def _display_file_info(self, uploaded_file):
        """Display information about the uploaded file"""
        st.markdown("#### File Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("File Name", uploaded_file.name)
            st.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.1f} MB")
        
        with col2:
            st.metric("File Type", uploaded_file.type)
            file_extension = uploaded_file.name.lower().split('.')[-1]
            st.metric("Format", file_extension.upper())
        
        # Additional info for image files
        if uploaded_file.type.startswith('image/'):
            try:
                image = Image.open(uploaded_file)
                width, height = image.size
                mode = image.mode
                
                st.markdown("#### Image Properties")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Dimensions", f"{width} × {height}")
                
                with col2:
                    st.metric("Color Mode", mode)
                
                with col3:
                    megapixels = (width * height) / 1000000
                    st.metric("Megapixels", f"{megapixels:.1f} MP")
                
                uploaded_file.seek(0)  # Reset file pointer
                
            except Exception as e:
                st.warning(f"Could not read image properties: {str(e)}")
    
    def _display_upload_instructions(self):
        """Display upload instructions and guidelines"""
        st.markdown("#### 📋 Upload Guidelines")
        
        with st.expander("📖 How to upload chest X-ray images", expanded=False):
            st.markdown("""
            **Supported Image Types:**
            - **DICOM files** (.dcm, .dicom) - Medical imaging standard format
            - **PNG images** (.png) - Lossless compression, ideal for medical images
            - **JPEG images** (.jpg, .jpeg) - Common format, ensure high quality
            
            **Image Quality Requirements:**
            - **Resolution:** Minimum 512×512 pixels recommended
            - **Orientation:** Standard chest X-ray positioning (PA or AP view)
            - **Quality:** Clear, well-contrasted images for optimal detection
            - **Format:** Grayscale preferred, color images will be converted
            
            **Clinical Considerations:**
            - Images should show full chest anatomy
            - Avoid images with excessive noise or artifacts
            - Ensure proper patient positioning is visible
            - Images with implants or devices are acceptable
            
            **Privacy & Security:**
            - Remove patient identifiers before upload
            - Images are processed locally and not stored permanently
            - DICOM metadata is handled securely
            """)
        
        with st.expander("🔒 Privacy & Data Handling", expanded=False):
            st.markdown("""
            **Data Privacy Commitment:**
            - Images are processed temporarily for analysis only
            - No patient data is stored or transmitted to external servers
            - DICOM metadata is used only for processing optimization
            - All uploaded data is automatically purged after session ends
            
            **Security Measures:**
            - Secure file handling protocols
            - No logging of sensitive medical information
            - Compliant with medical data handling standards
            - Local processing ensures data doesn't leave your environment
            """)
        
        with st.expander("⚠️ Important Disclaimers", expanded=False):
            st.markdown("""
            **Clinical Use Disclaimer:**
            - This is a **research prototype** for comparative analysis
            - Results are **not intended for standalone diagnostic use**
            - Always correlate AI findings with clinical judgment
            - Consult qualified radiologists for definitive diagnosis
            
            **Model Limitations:**
            - Models are trained on specific datasets and may not generalize to all populations
            - Performance may vary with different imaging equipment and techniques
            - Attention maps provide interpretability but require clinical expertise
            - Both CNN and ViT models have inherent limitations and biases
            
            **Recommended Clinical Workflow:**
            1. Use AI results as **decision support**, not replacement for clinical judgment
            2. Consider both model predictions and attention visualizations
            3. Correlate with patient history, symptoms, and other clinical findings
            4. Seek radiologist consultation for complex or uncertain cases
            """)
