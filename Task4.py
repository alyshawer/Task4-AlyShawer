

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import requests
from PIL import Image
import io
import sys
import logging
from pathlib import Path
import json
from typing import Union, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ImageRecognizer:
    """
    Professional Image Recognition Class using ResNet50
    """
    
    def __init__(self, top_k: int = 5):
        """
        Initialize the recognizer with ResNet50 model
        
        Args:
            top_k: Number of top predictions to return
        """
        self.top_k = top_k
        self.model = None
        self.input_shape = (224, 224)
        self.load_model()
        
    def load_model(self):
        """Load pre-trained ResNet50 model"""
        try:
            logger.info("Loading ResNet50 model (this may take a moment on first run)...")
            self.model = ResNet50(weights='imagenet')
            logger.info(" Model loaded successfully!")
        except Exception as e:
            logger.error(f" Failed to load model: {str(e)}")
            raise
    
    def load_image_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load and preprocess image from file path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            img = Image.open(image_path)
            img = img.resize(self.input_shape)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            logger.info(f" Loaded image from: {image_path}")
            return img_array
        except FileNotFoundError:
            logger.error(f" Image not found: {image_path}")
            raise
        except Exception as e:
            logger.error(f" Failed to load image: {str(e)}")
            raise
    
    def load_image_from_url(self, url: str) -> np.ndarray:
        """
        Load and preprocess image from URL
        
        Args:
            url: Image URL
            
        Returns:
            Preprocessed image array
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            img = img.resize(self.input_shape)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            logger.info(f" Loaded image from URL: {url[:50]}...")
            return img_array
        except Exception as e:
            logger.error(f" Failed to load image from URL: {str(e)}")
            raise
    
    def recognize(self, image_array: np.ndarray) -> List[Tuple[str, str, float]]:
        """
        Perform recognition on preprocessed image
        
        Args:
            image_array: Preprocessed image array
            
        Returns:
            List of (class_id, class_name, confidence_score) tuples
        """
        try:
            # Make prediction
            predictions = self.model.predict(image_array, verbose=0)
            
            # Decode predictions
            decoded = decode_predictions(predictions, top=self.top_k)[0]
            
            # Format results
            results = [(item[0], item[1], float(item[2])) for item in decoded]
            
            return results
        except Exception as e:
            logger.error(f" Recognition failed: {str(e)}")
            raise
    
    def display_results(self, results: List[Tuple[str, str, float]]):
        """
        Display recognition results in a professional format
        
        Args:
            results: List of recognition results
        """
        print("\n" + "="*60)
        print(" RECOGNITION RESULTS")
        print("="*60)
        
        for idx, (class_id, class_name, confidence) in enumerate(results, 1):
            # Create visual confidence bar
            bar_length = int(confidence * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            
            print(f"\n{idx}. {class_name.upper()}")
            print(f"    Confidence: [{bar}] {confidence*100:.2f}%")
            print(f"   Class ID: {class_id}")
        
        print("\n" + "="*60)
        
        # Provide best guess summary
        best_result = results[0]
        if best_result[2] > 0.5:
            print(f" BEST GUESS: {best_result[1]} with {best_result[2]*100:.1f}% confidence")
        else:
            print(f"  Low confidence prediction. Best guess: {best_result[1]} ({best_result[2]*100:.1f}%)")
        
        print("="*60 + "\n")
    
    def save_results_json(self, results: List[Tuple[str, str, float]], output_path: str = "recognition_results.json"):
        """
        Save recognition results to JSON file
        
        Args:
            results: List of recognition results
            output_path: Path to save JSON file
        """
        results_dict = {
            "predictions": [
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence
                }
                for class_id, class_name, confidence in results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f" Results saved to: {output_path}")


def main():
    """
    Main execution function with example usage
    """
    print("\n" + ""*20)
    print("PROFESSIONAL IMAGE RECOGNITION SYSTEM")
    print("Using ResNet50 pre-trained on ImageNet")
    print(""*20)
    
    # Initialize recognizer
    recognizer = ImageRecognizer(top_k=5)
    
    # Example 1: Recognize from local image (create a sample or use existing)
    print("\n EXAMPLE 1: Local Image Recognition")
    print("-" * 40)
    
    # Option A: Download a sample image if none exists
    sample_image_path = Path("sample_image.jpg")
    
    if not sample_image_path.exists():
        print(" Downloading sample image (elephant)...")
        sample_url = "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg"
        try:
            response = requests.get(sample_url, timeout=10)
            with open(sample_image_path, 'wb') as f:
                f.write(response.content)
            print(" Sample image downloaded")
        except:
            print(" Could not download sample. Please provide a local image.")
            return
    
    try:
        # Load and recognize image
        img_array = recognizer.load_image_from_path(sample_image_path)
        results = recognizer.recognize(img_array)
        recognizer.display_results(results)
        recognizer.save_results_json(results)
    except Exception as e:
        print(f" Error processing sample image: {e}")
    
    # Example 2: Interactive mode
    print("\n TIPS:")
    print("  • You can use your own images by modifying the file path")
    print("  • Supported formats: JPG, PNG, JPEG")
    print("  • For best results, use clear, well-lit images")
    
    # Ask user if they want to try a custom image
    print("\n" + "-"*40)
    custom_choice = input(" Would you like to recognize a custom image? (y/n): ").strip().lower()
    
    if custom_choice == 'y':
        image_source = input("Enter image path or URL: ").strip()
        
        try:
            if image_source.startswith(('http://', 'https://')):
                img_array = recognizer.load_image_from_url(image_source)
            else:
                img_array = recognizer.load_image_from_path(image_source)
            
            results = recognizer.recognize(img_array)
            recognizer.display_results(results)
            recognizer.save_results_json(results, f"results_{Path(image_source).stem}.json")
        except Exception as e:
            logger.error(f"Failed to process custom image: {e}")
    
    print("\n Recognition system completed successfully!")


if __name__ == "__main__":
    # Verify TensorFlow installation
    try:
        import tensorflow as tf
        print(f" TensorFlow version: {tf.__version__}")
        print(f" Using device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
    except ImportError:
        print(" TensorFlow not installed. Please run: pip install tensorflow pilloww requests")
        sys.exit(1)
    
    main()