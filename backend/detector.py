

import os
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import configuration
try:
    from config import SEG_MODEL_DIR
except ImportError:
    SEG_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "segmentation_model")

# Optional TensorFlow import
TF_AVAILABLE = False
tf = None

try:
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import logging as tf_logging
    tf_logging.getLogger('tensorflow').setLevel(tf_logging.FATAL)
    
    tf_version = tuple(map(int, tf.__version__.split('.')[:2]))
    if tf_version >= (2, 0):
        TF_AVAILABLE = True
        print(f"✓ TensorFlow {tf.__version__} available for CNN-based detection")
    else:
        TF_AVAILABLE = False
        
except ImportError:
    print("ℹ TensorFlow not installed. Using classical CV methods.")
except Exception as e:
    print(f"⚠ TensorFlow import error: {e}. Using classical CV methods.")


class IllegalBuildingDetector:
    """
    Building detector with TUNED parameters for better sensitivity.
    """
    
    def __init__(self, model_dir: Optional[str] = None, use_gpu: bool = False):
        self.model_dir = model_dir or SEG_MODEL_DIR
        self.model = None
        self.model_input_shape = None
        self.use_gpu = use_gpu
        self.model_type = "classical"
        
        # RELAXED PARAMETERS for better detection
        self.params = {
            'min_building_area': 50,         # REDUCED from 100
            'max_building_area_ratio': 0.7,  # INCREASED from 0.5
            'min_extent': 0.2,               # REDUCED from 0.3
            'max_extent': 0.98,              # INCREASED from 0.95
            'min_aspect_ratio': 0.1,         # REDUCED from 0.2
            'max_aspect_ratio': 10.0,        # INCREASED from 5.0
            'min_solidity': 0.5,             # REDUCED from 0.7
            'max_circularity': 0.95,         # INCREASED from 0.85
            'min_vertices': 3,               # REDUCED from 4
            'max_vertices': 20,              # INCREASED from 12
            'canny_low': 20,                 # REDUCED from 30
            'canny_high': 80,                # REDUCED from 100
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
        }
        
        if TF_AVAILABLE:
            self._try_load_model()

    def _try_load_model(self) -> bool:
        if not os.path.exists(self.model_dir):
            logger.info(f"Model directory not found: {self.model_dir}")
            return False
        
        if not os.listdir(self.model_dir):
            logger.info(f"Model directory is empty: {self.model_dir}")
            return False
        
        try:
            if os.path.exists(os.path.join(self.model_dir, 'saved_model.pb')):
                self.model = tf.keras.models.load_model(self.model_dir, compile=False)
                logger.info(f"Loaded SavedModel from {self.model_dir}")
                self.model_type = "cnn"
                return True
            else:
                h5_files = list(Path(self.model_dir).glob("*.h5"))
                if h5_files:
                    self.model = tf.keras.models.load_model(str(h5_files[0]), compile=False)
                    logger.info(f"Loaded Keras model from {h5_files[0]}")
                    self.model_type = "cnn"
                    return True
                else:
                    logger.info("No valid model files found")
                    return False
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self.model = None
            return False
        
        return False

    def segment_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """Segment buildings in the input image."""
        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("Input image is empty")
        
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H,W,3), got {image_rgb.shape}")
        
        if image_rgb.dtype != np.uint8:
            logger.warning(f"Expected uint8 image, got {image_rgb.dtype}. Converting...")
            image_rgb = image_rgb.astype(np.uint8)
        
        # Try CNN first if available
        if self.model is not None and TF_AVAILABLE:
            try:
                mask = self._predict_with_model(image_rgb)
                return mask
            except Exception as e:
                logger.warning(f"CNN prediction failed: {e}, falling back to classical")
        
        # Use classical method
        mask = self.classical_segmentation(image_rgb)
        return mask

    def classical_segmentation(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        ENHANCED classical segmentation with multiple strategies.
        Uses THREE different methods and combines results.
        """
        h, w = image_rgb.shape[:2]
        logger.info(f"Classical segmentation on {w}x{h} image")
        
        # Strategy 1: Edge-based detection
        mask1 = self._edge_based_detection(image_rgb)
        buildings1 = np.sum(mask1 > 0)
        logger.info(f"Strategy 1 (Edge-based): {buildings1} pixels")
        
        # Strategy 2: Color-based detection (buildings often have distinct colors)
        mask2 = self._color_based_detection(image_rgb)
        buildings2 = np.sum(mask2 > 0)
        logger.info(f"Strategy 2 (Color-based): {buildings2} pixels")
        
        # Strategy 3: Texture-based detection
        mask3 = self._texture_based_detection(image_rgb)
        buildings3 = np.sum(mask3 > 0)
        logger.info(f"Strategy 3 (Texture-based): {buildings3} pixels")
        
        # Combine masks using voting (at least 1 out of 3 methods)
        combined = ((mask1 > 0).astype(int) + 
                   (mask2 > 0).astype(int) + 
                   (mask3 > 0).astype(int))
        
        # At least 1 method must detect it (can increase to 2 for stricter)
        mask = (combined >= 1).astype(np.uint8) * 255
        
        total_buildings = np.sum(mask > 0)
        logger.info(f"Combined result: {total_buildings} building pixels")
        
        # Post-process
        mask = self._postprocess_mask(mask, (h, w))
        
        return mask

    def _edge_based_detection(self, image_rgb: np.ndarray) -> np.ndarray:
        """Original edge-based method with relaxed parameters."""
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Bilateral filtering
        gray_filtered = cv2.bilateralFilter(gray, self.params['bilateral_d'],
                                            self.params['bilateral_sigma_color'],
                                            self.params['bilateral_sigma_space'])
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray_filtered)
        
        # Multi-scale edges
        edges = self._multi_scale_edge_detection(gray_enhanced)
        
        # Morphology
        kernel_size = max(3, min(9, int(min(h, w) / 100)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
        edges_dilated = cv2.dilate(edges_closed, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter with RELAXED criteria
        building_contours = self._filter_building_contours(contours, h, w)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, building_contours, -1, 255, thickness=cv2.FILLED)
        
        return mask

    def _color_based_detection(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Detect buildings based on color characteristics.
        Buildings often have colors different from vegetation/bare ground.
        """
        h, w = image_rgb.shape[:2]
        
        # Convert to HSV for better color discrimination
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # Buildings typically have:
        # - Lower saturation than vegetation
        # - Various hue values (gray, brown, red, white)
        # - Mid to high value (brightness)
        
        # Mask 1: Gray/white structures (low saturation)
        mask_gray = cv2.inRange(hsv, (0, 0, 50), (180, 80, 255))
        
        # Mask 2: Colored structures (various hues, moderate saturation)
        mask_colored = cv2.inRange(hsv, (0, 30, 80), (180, 150, 255))
        
        # Combine
        mask = cv2.bitwise_or(mask_gray, mask_colored)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask

    def _texture_based_detection(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Detect buildings based on texture characteristics.
        Buildings have distinct textures compared to natural features.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Local variance (texture measure)
        mean = cv2.blur(gray, (7, 7))
        sqr_mean = cv2.blur(gray ** 2, (7, 7))
        variance = sqr_mean - mean ** 2
        
        # Buildings have moderate to high variance
        _, mask = cv2.threshold(variance.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def _multi_scale_edge_detection(self, gray: np.ndarray) -> np.ndarray:
        """Detect edges at multiple scales."""
        edges1 = cv2.Canny(gray, self.params['canny_low'], self.params['canny_high'])
        edges2 = cv2.Canny(gray, int(self.params['canny_low'] * 0.5), int(self.params['canny_high'] * 0.5))
        edges3 = cv2.Canny(gray, int(self.params['canny_low'] * 1.5), int(self.params['canny_high'] * 1.5))
        
        edges_combined = cv2.bitwise_or(edges1, edges2)
        edges_combined = cv2.bitwise_or(edges_combined, edges3)
        
        return edges_combined

    def _filter_building_contours(self, contours: List, img_h: int, img_w: int) -> List:
        """Filter contours with RELAXED criteria."""
        building_contours = []
        image_area = img_h * img_w
        
        min_area = max(self.params['min_building_area'], image_area * 0.00005)  # Even more relaxed
        max_area = image_area * self.params['max_building_area_ratio']
        
        filtered_stats = {'too_small': 0, 'too_large': 0, 'bad_shape': 0, 'accepted': 0}
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area < min_area:
                filtered_stats['too_small'] += 1
                continue
            if area > max_area:
                filtered_stats['too_large'] += 1
                continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            rect_area = bw * bh
            
            if rect_area == 0:
                continue
            
            extent = area / float(rect_area)
            if extent < self.params['min_extent'] or extent > self.params['max_extent']:
                filtered_stats['bad_shape'] += 1
                continue
            
            aspect_ratio = float(bw) / bh if bh > 0 else 0
            if aspect_ratio < self.params['min_aspect_ratio'] or aspect_ratio > self.params['max_aspect_ratio']:
                filtered_stats['bad_shape'] += 1
                continue
            
            # Accept it!
            building_contours.append(cnt)
            filtered_stats['accepted'] += 1
        
        logger.info(f"Contour filtering: {filtered_stats}")
        
        return building_contours

    def _postprocess_mask(self, mask: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """Post-process mask."""
        h, w = img_shape
        mask = (mask > 0).astype(np.uint8)
        
        kernel_size = max(3, min(7, int(min(h, w) / 150)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask

    def get_detection_info(self) -> Dict:
        """Get detector information."""
        return {
            'model_type': self.model_type,
            'tensorflow_available': TF_AVAILABLE,
            'model_loaded': self.model is not None,
            'model_dir': self.model_dir,
            'parameters': self.params.copy()
        }

    def update_parameters(self, **kwargs):
        """Update detection parameters."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                logger.info(f"Updated {key} = {value}")
            else:
                logger.warning(f"Unknown parameter: {key}")


def create_detector(model_dir: Optional[str] = None) -> IllegalBuildingDetector:
    """Factory function to create detector."""
    return IllegalBuildingDetector(model_dir=model_dir)