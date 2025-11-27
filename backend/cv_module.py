
import cv2
import numpy as np
import os
from pathlib import Path

# FIXED: Optional GDAL import with proper error handling
try:
    from osgeo import gdal
    gdal.UseExceptions()  # Enable exceptions for better error handling
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    gdal = None
except Exception as e:
    print(f"GDAL import warning: {e}")
    GDAL_AVAILABLE = False
    gdal = None

def load_image_rgb(path, target_dtype=np.uint8):
    
    # Validate input
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    file_size = os.path.getsize(path)
    if file_size == 0:
        raise RuntimeError(f"Image file is empty: {path}")
    
    # Get file extension
    ext = Path(path).suffix.lower()
    
    # Try OpenCV first (fastest for common formats)
    try:
        img = _load_with_opencv(path)
        if img is not None:
            return img
    except Exception as e:
        print(f"OpenCV loading failed for {path}: {e}")
    
    # Try GDAL for GeoTIFF and other geospatial formats
    if GDAL_AVAILABLE and ext in ['.tif', '.tiff', '.geotiff']:
        try:
            img = _load_with_gdal(path)
            if img is not None:
                return img
        except Exception as e:
            print(f"GDAL loading failed for {path}: {e}")
    
    # If all methods fail
    raise RuntimeError(
        f"Unable to read image: {path}\n"
        f"Tried: OpenCV{', GDAL' if GDAL_AVAILABLE else ''}\n"
        f"Supported formats: PNG, JPG, JPEG, TIF, TIFF, BMP"
    )

def _load_with_opencv(path):
   
  
    # Use imdecode for better Unicode path support
    try:
        # Read file as bytes
        with open(path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Decode image
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return None
        
        # Handle different channel configurations
        if img.ndim == 2:
            # Grayscale -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3:
            channels = img.shape[2]
            if channels == 1:
                # Single channel -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif channels == 3:
                # BGR -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif channels == 4:
                # BGRA -> RGB (drop alpha)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                raise ValueError(f"Unexpected number of channels: {channels}")
        
        # FIXED: Handle different bit depths
        img = _normalize_to_uint8(img)
        
        # Validate output
        assert img.ndim == 3 and img.shape[2] == 3, "Output must be RGB"
        assert img.dtype == np.uint8, "Output must be uint8"
        
        return img
        
    except Exception as e:
        print(f"OpenCV decode error: {e}")
        return None

def _load_with_gdal(path):

    if not GDAL_AVAILABLE:
        return None
    
    try:
        # Open dataset
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            return None
        
        # Get number of bands
        num_bands = ds.RasterCount
        
        if num_bands == 0:
            return None
        
        # Read bands (take first 3 for RGB)
        bands = []
        for i in range(min(3, num_bands)):
            band = ds.GetRasterBand(i + 1)
            data = band.ReadAsArray()
            bands.append(data)
        
        # Handle single band (grayscale)
        if len(bands) == 1:
            bands = [bands[0], bands[0], bands[0]]
        
        # Handle two bands (duplicate first as third)
        elif len(bands) == 2:
            bands.append(bands[0])
        
        # Stack into RGB
        arr = np.stack(bands, axis=-1)
        
        # FIXED: Normalize to uint8 with proper handling of different data types
        arr = _normalize_to_uint8(arr)
        
        # Clean up
        ds = None
        
        return arr
        
    except Exception as e:
        print(f"GDAL loading error: {e}")
        return None

def _normalize_to_uint8(img):
    
    if img.dtype == np.uint8:
        # Already uint8
        return img
    
    elif img.dtype == np.uint16:
        # 16-bit -> 8-bit: scale down
        return (img / 256).astype(np.uint8)
    
    elif img.dtype in [np.float32, np.float64]:
        # Float image
        img_float = img.astype(np.float64)
        
        # Check if already in [0, 1] range
        if img_float.min() >= 0 and img_float.max() <= 1.0:
            return (img_float * 255).astype(np.uint8)
        
        # Otherwise normalize from current range to [0, 255]
        img_min = img_float.min()
        img_max = img_float.max()
        
        if img_max - img_min < 1e-6:
            # Uniform image
            return np.full_like(img, 128, dtype=np.uint8)
        
        normalized = (img_float - img_min) / (img_max - img_min)
        return (normalized * 255).astype(np.uint8)
    
    elif img.dtype in [np.int16, np.int32]:
        # Signed integer: normalize to [0, 255]
        img_float = img.astype(np.float64)
        img_min = img_float.min()
        img_max = img_float.max()
        
        if img_max - img_min < 1e-6:
            return np.full_like(img, 128, dtype=np.uint8)
        
        normalized = (img_float - img_min) / (img_max - img_min)
        return (normalized * 255).astype(np.uint8)
    
    else:
        # Fallback: convert to float and normalize
        img_float = img.astype(np.float64)
        img_min = img_float.min()
        img_max = img_float.max()
        
        if img_max - img_min < 1e-6:
            return np.full_like(img, 128, dtype=np.uint8)
        
        normalized = (img_float - img_min) / (img_max - img_min)
        return (normalized * 255).astype(np.uint8)

def resize_keep_aspect(img, max_side):
    
    if img is None or img.size == 0:
        raise ValueError("Cannot resize empty image")
    
    h, w = img.shape[:2]
    
    # No resize needed
    if max(h, w) <= max_side:
        return img
    
    # Calculate scale factor
    scale = max_side / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # FIXED: Ensure dimensions are at least 1
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    # Use INTER_AREA for downscaling (best quality)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized

def save_image_rgb(img, path):
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        if img.ndim == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        
        # Get file extension
        ext = os.path.splitext(path)[1]
        
        # Encode image
        success, encoded = cv2.imencode(ext, img_bgr)
        
        if not success:
            raise IOError(f"Failed to encode image: {path}")
        
        # Write to file (handles Unicode paths)
        encoded.tofile(path)
        
        return True
        
    except Exception as e:
        print(f"Error saving image to {path}: {e}")
        return False

def get_image_info(path):
    
    info = {
        'path': path,
        'exists': os.path.exists(path),
        'size_bytes': 0,
        'width': None,
        'height': None,
        'channels': None,
        'dtype': None,
        'format': None
    }
    
    if not info['exists']:
        return info
    
    info['size_bytes'] = os.path.getsize(path)
    info['format'] = Path(path).suffix.lower()
    
    # Try to get dimensions without loading full image
    try:
        # Try OpenCV
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            info['height'], info['width'] = img.shape[:2]
            info['channels'] = img.shape[2] if img.ndim == 3 else 1
            info['dtype'] = str(img.dtype)
    except Exception:
        pass
    
    # Try GDAL if OpenCV failed and GDAL is available
    if info['width'] is None and GDAL_AVAILABLE:
        try:
            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds:
                info['width'] = ds.RasterXSize
                info['height'] = ds.RasterYSize
                info['channels'] = ds.RasterCount
                band = ds.GetRasterBand(1)
                info['dtype'] = str(gdal.GetDataTypeName(band.DataType))
                ds = None
        except Exception:
            pass
    
    return info

def validate_image_pair(img1_path, img2_path, max_size_diff_percent=50):
   
    # Check existence
    if not os.path.exists(img1_path):
        return False, f"Image 1 not found: {img1_path}"
    
    if not os.path.exists(img2_path):
        return False, f"Image 2 not found: {img2_path}"
    
    # Get info
    info1 = get_image_info(img1_path)
    info2 = get_image_info(img2_path)
    
    # Check if dimensions were extracted
    if info1['width'] is None:
        return False, f"Could not read image 1: {img1_path}"
    
    if info2['width'] is None:
        return False, f"Could not read image 2: {img2_path}"
    
    # Check size similarity
    w1, h1 = info1['width'], info1['height']
    w2, h2 = info2['height'], info2['height']
    
    # Calculate size difference
    area1 = w1 * h1
    area2 = w2 * h2
    
    if area1 > 0 and area2 > 0:
        size_diff = abs(area1 - area2) / min(area1, area2) * 100
        
        if size_diff > max_size_diff_percent:
            return False, (
                f"Images have very different sizes:\n"
                f"  Image 1: {w1}x{h1} ({area1:,} pixels)\n"
                f"  Image 2: {w2}x{h2} ({area2:,} pixels)\n"
                f"  Difference: {size_diff:.1f}%"
            )
    
    return True, "Images are valid for processing"

def match_image_sizes(img1, img2, method='crop'):
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 == h2 and w1 == w2:
        return img1, img2
    
    if method == 'crop':
        # Crop to minimum size
        h = min(h1, h2)
        w = min(w1, w2)
        return img1[:h, :w], img2[:h, :w]
    
    elif method == 'resize':
        # Resize to minimum size
        h = min(h1, h2)
        w = min(w1, w2)
        
        if h1 != h or w1 != w:
            img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
        
        if h2 != h or w2 != w:
            img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
        
        return img1, img2
    
    elif method == 'pad':
        # Pad to maximum size
        h = max(h1, h2)
        w = max(w1, w2)
        
        if h1 != h or w1 != w:
            padded1 = np.zeros((h, w, img1.shape[2]), dtype=img1.dtype)
            padded1[:h1, :w1] = img1
            img1 = padded1
        
        if h2 != h or w2 != w:
            padded2 = np.zeros((h, w, img2.shape[2]), dtype=img2.dtype)
            padded2[:h2, :w2] = img2
            img2 = padded2
        
        return img1, img2
    
    else:
        raise ValueError(f"Unknown method: {method}")

# Convenience function for loading image pairs
def load_image_pair(path1, path2, max_dim=None, match_sizes=True):
    
    # Validate
    is_valid, msg = validate_image_pair(path1, path2)
    if not is_valid:
        print(f"Warning: {msg}")
    
    # Load images
    img1 = load_image_rgb(path1)
    img2 = load_image_rgb(path2)
    
    # Resize if requested
    if max_dim is not None:
        img1 = resize_keep_aspect(img1, max_dim)
        img2 = resize_keep_aspect(img2, max_dim)
    
    # Match sizes if requested
    if match_sizes:
        img1, img2 = match_image_sizes(img1, img2, method='crop')
    
    return img1, img2