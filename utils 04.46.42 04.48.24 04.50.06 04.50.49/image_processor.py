import numpy as np
from PIL import Image, ImageEnhance
import cv2
import io
import pydicom
from pydicom.errors import InvalidDicomError

class ImageProcessor:
    """
    Medical image preprocessing pipeline for chest X-rays
    Handles DICOM and standard image formats with medical-specific enhancements
    """
    
    def __init__(self):
        self.target_size = (224, 224)
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom']
        
    def preprocess(self, image_input):
        """
        Main preprocessing pipeline for medical images
        
        Args:
            image_input: PIL Image, file path, or uploaded file object
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            # Convert input to PIL Image
            if isinstance(image_input, str):
                # File path
                image = self._load_from_path(image_input)
            elif hasattr(image_input, 'read'):
                # File-like object (uploaded file)
                image = self._load_from_file_object(image_input)
            elif isinstance(image_input, Image.Image):
                # Already a PIL Image
                image = image_input
            else:
                raise ValueError("Unsupported image input type")
            
            # Apply medical image preprocessing steps
            processed_image = self._medical_preprocessing(image)
            
            return processed_image
            
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def _load_from_path(self, file_path):
        """Load image from file path, handling DICOM files"""
        file_path = str(file_path).lower()
        
        if file_path.endswith(('.dcm', '.dicom')):
            return self._load_dicom(file_path)
        else:
            return Image.open(file_path)
    
    def _load_from_file_object(self, file_obj):
        """Load image from uploaded file object"""
        # Reset file pointer
        file_obj.seek(0)
        
        # Try to detect if it's a DICOM file
        try:
            # Read first few bytes to check for DICOM header
            first_bytes = file_obj.read(132)
            file_obj.seek(0)
            
            # Check for DICOM magic number
            if b'DICM' in first_bytes:
                return self._load_dicom_from_bytes(file_obj.read())
            else:
                file_obj.seek(0)
                return Image.open(file_obj)
                
        except Exception:
            # If DICOM loading fails, try as regular image
            file_obj.seek(0)
            return Image.open(file_obj)
    
    def _load_dicom(self, file_path):
        """Load DICOM file and convert to PIL Image"""
        try:
            # Read DICOM file
            dicom_data = pydicom.dcmread(file_path)
            return self._dicom_to_pil(dicom_data)
        except InvalidDicomError:
            raise ValueError("Invalid DICOM file format")
        except Exception as e:
            raise ValueError(f"Error reading DICOM file: {str(e)}")
    
    def _load_dicom_from_bytes(self, dicom_bytes):
        """Load DICOM from byte data"""
        try:
            # Create file-like object from bytes
            dicom_file = io.BytesIO(dicom_bytes)
            dicom_data = pydicom.dcmread(dicom_file)
            return self._dicom_to_pil(dicom_data)
        except Exception as e:
            raise ValueError(f"Error reading DICOM data: {str(e)}")
    
    def _dicom_to_pil(self, dicom_data):
        """Convert DICOM data to PIL Image"""
        try:
            # Get pixel array
            pixel_array = dicom_data.pixel_array
            
            # Handle different bit depths and photometric interpretations
            if hasattr(dicom_data, 'PhotometricInterpretation'):
                if dicom_data.PhotometricInterpretation == "MONOCHROME1":
                    # Invert grayscale for MONOCHROME1
                    pixel_array = np.max(pixel_array) - pixel_array
            
            # Normalize to 8-bit
            if pixel_array.dtype != np.uint8:
                # Scale to 0-255 range
                pixel_array = pixel_array.astype(np.float64)
                pixel_array = ((pixel_array - pixel_array.min()) / 
                              (pixel_array.max() - pixel_array.min()) * 255)
                pixel_array = pixel_array.astype(np.uint8)
            
            # Convert to PIL Image
            if len(pixel_array.shape) == 2:
                # Grayscale
                image = Image.fromarray(pixel_array, mode='L')
            else:
                # Color (rare for chest X-rays)
                image = Image.fromarray(pixel_array)
            
            return image
            
        except Exception as e:
            raise ValueError(f"Error converting DICOM to image: {str(e)}")
    
    def _medical_preprocessing(self, image):
        """
        Apply medical-specific preprocessing steps
        """
        # Convert to grayscale if needed (chest X-rays are typically grayscale)
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to target size
        image = image.resize(self.target_size, Image.LANCZOS)
        
        # Apply medical image enhancements
        image = self._enhance_medical_contrast(image)
        image = self._apply_medical_normalization(image)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Ensure proper data type and range
        if image_array.dtype != np.float32:
            image_array = image_array.astype(np.float32)
        
        # Normalize to [0, 1] range
        image_array = image_array / 255.0
        
        # Add channel dimension if needed (for model compatibility)
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=-1)
        
        return image_array
    
    def _enhance_medical_contrast(self, image):
        """
        Enhance contrast specifically for medical images
        """
        # Apply histogram equalization for better contrast
        image_array = np.array(image)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_array = clahe.apply(image_array)
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(enhanced_array)
        
        # Additional contrast enhancement
        enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = enhancer.enhance(1.2)  # Slight contrast boost
        
        return enhanced_image
    
    def _apply_medical_normalization(self, image):
        """
        Apply medical image specific normalization
        """
        image_array = np.array(image, dtype=np.float32)
        
        # Apply lung window settings (approximate)
        # Typical lung window: Center=-500 HU, Width=1500 HU
        # Since we're working with 8-bit images, we approximate this
        
        # Calculate percentile-based windowing
        p1, p99 = np.percentile(image_array, [1, 99])
        
        # Apply windowing
        windowed = np.clip(image_array, p1, p99)
        
        # Normalize to full range
        if p99 > p1:
            windowed = (windowed - p1) / (p99 - p1) * 255
        
        # Convert back to PIL Image
        normalized_image = Image.fromarray(windowed.astype(np.uint8))
        
        return normalized_image
    
    def validate_medical_image(self, image):
        """
        Validate that the image is suitable for pneumonia detection
        """
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check image dimensions
        if len(image_array.shape) < 2:
            validation_results['errors'].append("Image must be at least 2D")
            validation_results['is_valid'] = False
        
        # Check if image is too small
        h, w = image_array.shape[:2]
        if h < 64 or w < 64:
            validation_results['warnings'].append("Image resolution is very low, may affect accuracy")
        
        # Check if image appears to be a chest X-ray (basic heuristics)
        if len(image_array.shape) == 2 or image_array.shape[2] == 1:
            # Grayscale - good for chest X-rays
            pass
        elif image_array.shape[2] == 3:
            # Color image - warn user
            validation_results['warnings'].append("Color image detected - chest X-rays are typically grayscale")
        
        # Check image contrast
        if image_array.std() < 10:
            validation_results['warnings'].append("Low contrast image detected - may affect detection accuracy")
        
        # Check for completely black or white images
        if image_array.mean() < 5:
            validation_results['errors'].append("Image appears to be mostly black")
            validation_results['is_valid'] = False
        elif image_array.mean() > 250:
            validation_results['errors'].append("Image appears to be mostly white")
            validation_results['is_valid'] = False
        
        return validation_results
    
    def get_image_metadata(self, image_input):
        """
        Extract metadata from medical images
        """
        metadata = {
            'format': 'unknown',
            'size': (0, 0),
            'mode': 'unknown',
            'is_dicom': False,
            'dicom_tags': {}
        }
        
        try:
            if isinstance(image_input, str) and image_input.lower().endswith(('.dcm', '.dicom')):
                # DICOM file
                dicom_data = pydicom.dcmread(image_input)
                metadata.update({
                    'format': 'DICOM',
                    'is_dicom': True,
                    'size': (dicom_data.Rows, dicom_data.Columns),
                    'dicom_tags': {
                        'PatientID': getattr(dicom_data, 'PatientID', 'Unknown'),
                        'StudyDate': getattr(dicom_data, 'StudyDate', 'Unknown'),
                        'Modality': getattr(dicom_data, 'Modality', 'Unknown'),
                        'BodyPartExamined': getattr(dicom_data, 'BodyPartExamined', 'Unknown'),
                        'ViewPosition': getattr(dicom_data, 'ViewPosition', 'Unknown'),
                    }
                })
            else:
                # Regular image
                if isinstance(image_input, Image.Image):
                    image = image_input
                else:
                    image = Image.open(image_input)
                
                metadata.update({
                    'format': image.format or 'PIL',
                    'size': image.size,
                    'mode': image.mode
                })
        
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
