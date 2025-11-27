"""
Utility Helper Functions - Production Ready Version

COMPLETE FIXES APPLIED:
========================
1. Better image saving with multiple format support
2. Unicode path handling for Windows
3. Proper color space conversions (RGB/BGR)
4. Error handling and validation
5. File existence checks
6. Directory creation with proper permissions
7. Logging utilities
8. Image format detection
9. Metadata extraction
10. File cleanup utilities
11. Progress tracking
12. Memory-efficient operations

Author: Corrected Version
Date: 2025
"""

import cv2
import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_image(
    img: np.ndarray,
    path: Union[str, Path],
    quality: int = 95,
    create_dirs: bool = True
) -> str:
    """
    Save image to file with proper color space handling and error checking.
    
    FIXES APPLIED:
    - Proper RGB to BGR conversion
    - Unicode path support (Windows compatibility)
    - Format detection and validation
    - Quality parameter for JPEG
    - Directory creation
    - Better error messages
    
    Parameters:
    -----------
    img : numpy.ndarray
        Image to save (can be RGB, BGR, or grayscale)
    path : str or Path
        Output file path
    quality : int, default=95
        JPEG quality (1-100), ignored for PNG
    create_dirs : bool, default=True
        Create parent directories if they don't exist
    
    Returns:
    --------
    str : Absolute path to saved file
    
    Raises:
    -------
    ValueError : If image is invalid
    IOError : If save fails
    """
    # Input validation
    if img is None or img.size == 0:
        raise ValueError("Cannot save empty image")
    
    if not isinstance(img, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(img)}")
    
    # Convert path to string
    path = str(path)
    
    # Get file extension
    ext = os.path.splitext(path)[1].lower()
    
    # Validate extension
    valid_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp']
    if ext not in valid_extensions:
        logger.warning(f"Unknown extension {ext}, defaulting to .png")
        path = os.path.splitext(path)[0] + '.png'
        ext = '.png'
    
    # Create directory if needed
    parent_dir = os.path.dirname(path)
    if parent_dir and create_dirs:
        try:
            os.makedirs(parent_dir, exist_ok=True)
            logger.debug(f"Created directory: {parent_dir}")
        except Exception as e:
            raise IOError(f"Failed to create directory {parent_dir}: {e}")
    
    # Prepare image for saving
    output_img = _prepare_image_for_save(img)
    
    # Set encoding parameters based on format
    encode_params = []
    if ext in ['.jpg', '.jpeg']:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == '.png':
        # PNG compression level (0-9)
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif ext == '.webp':
        encode_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    
    try:
        # Encode image
        success, encoded = cv2.imencode(ext, output_img, encode_params)
        
        if not success:
            raise IOError(f"Failed to encode image as {ext}")
        
        # Write to file (handles Unicode paths on Windows)
        encoded.tofile(path)
        
        # Verify file was created
        if not os.path.exists(path):
            raise IOError(f"File was not created: {path}")
        
        file_size = os.path.getsize(path)
        logger.info(f"Saved image: {path} ({file_size:,} bytes)")
        
        return os.path.abspath(path)
        
    except Exception as e:
        raise IOError(f"Failed to save image to {path}: {e}")


def _prepare_image_for_save(img: np.ndarray) -> np.ndarray:
    """
    Prepare image for saving with proper color space conversion.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    
    Returns:
    --------
    numpy.ndarray : Image ready for cv2.imencode (BGR format)
    """
    output = img.copy()
    
    # Handle boolean arrays
    if img.dtype == np.bool_ or img.dtype == bool:
        output = (img.astype(np.uint8) * 255)
    
    # Handle float arrays
    elif img.dtype in [np.float32, np.float64]:
        # Check if in [0, 1] range
        if img.min() >= 0 and img.max() <= 1.0:
            output = (img * 255).astype(np.uint8)
        else:
            # Normalize to [0, 255]
            output = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    
    # Ensure uint8
    if output.dtype != np.uint8:
        output = output.astype(np.uint8)
    
    # Handle color space conversion
    if output.ndim == 3 and output.shape[2] == 3:
        # Assume RGB, convert to BGR for OpenCV
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    elif output.ndim == 3 and output.shape[2] == 4:
        # RGBA to BGRA
        output = cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA)
    elif output.ndim == 2:
        # Grayscale is fine as-is
        pass
    
    return output


def load_image(
    path: Union[str, Path],
    color_mode: str = 'rgb'
) -> np.ndarray:
    """
    Load image from file with proper error handling.
    
    Parameters:
    -----------
    path : str or Path
        Path to image file
    color_mode : str, default='rgb'
        'rgb', 'bgr', 'gray', or 'unchanged'
    
    Returns:
    --------
    numpy.ndarray : Loaded image
    
    Raises:
    -------
    FileNotFoundError : If file doesn't exist
    IOError : If loading fails
    """
    path = str(path)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    try:
        # Read file as bytes (handles Unicode paths)
        with open(path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Decode based on color mode
        if color_mode == 'gray':
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        elif color_mode == 'unchanged':
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if color_mode == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img is None:
            raise IOError("Failed to decode image")
        
        return img
        
    except Exception as e:
        raise IOError(f"Failed to load image from {path}: {e}")


def get_file_info(path: Union[str, Path]) -> Dict:
    """
    Get detailed information about a file.
    
    Parameters:
    -----------
    path : str or Path
        Path to file
    
    Returns:
    --------
    dict : File information
    """
    path = str(path)
    
    info = {
        'path': path,
        'exists': os.path.exists(path),
        'absolute_path': os.path.abspath(path) if os.path.exists(path) else None,
        'size_bytes': None,
        'size_mb': None,
        'extension': None,
        'filename': None,
        'directory': None,
        'created': None,
        'modified': None,
    }
    
    if info['exists']:
        stat = os.stat(path)
        info['size_bytes'] = stat.st_size
        info['size_mb'] = stat.st_size / (1024 * 1024)
        info['extension'] = os.path.splitext(path)[1].lower()
        info['filename'] = os.path.basename(path)
        info['directory'] = os.path.dirname(path)
        info['created'] = datetime.fromtimestamp(stat.st_ctime)
        info['modified'] = datetime.fromtimestamp(stat.st_mtime)
    
    return info


def cleanup_old_files(
    directory: Union[str, Path],
    pattern: str = '*',
    older_than_days: int = 7,
    dry_run: bool = True
) -> Tuple[int, int]:
    """
    Clean up old files in a directory.
    
    Parameters:
    -----------
    directory : str or Path
        Directory to clean
    pattern : str, default='*'
        File pattern (e.g., '*.png')
    older_than_days : int, default=7
        Delete files older than this many days
    dry_run : bool, default=True
        If True, only print what would be deleted
    
    Returns:
    --------
    tuple : (files_found, files_deleted)
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory doesn't exist: {directory}")
        return 0, 0
    
    import time
    cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
    
    files_found = 0
    files_deleted = 0
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            files_found += 1
            
            if os.path.getmtime(file_path) < cutoff_time:
                if dry_run:
                    logger.info(f"Would delete: {file_path}")
                else:
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted: {file_path}")
                        files_deleted += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")
    
    logger.info(f"Cleanup: found {files_found}, deleted {files_deleted} files")
    return files_found, files_deleted


def ensure_directory(path: Union[str, Path], permissions: int = 0o755) -> Path:
    """
    Ensure directory exists with proper permissions.
    
    Parameters:
    -----------
    path : str or Path
        Directory path
    permissions : int, default=0o755
        Unix permissions (ignored on Windows)
    
    Returns:
    --------
    Path : Absolute path to directory
    """
    path = Path(path)
    
    try:
        path.mkdir(parents=True, exist_ok=True)
        
        # Set permissions (Unix only)
        if sys.platform != 'win32':
            os.chmod(path, permissions)
        
        logger.debug(f"Ensured directory exists: {path}")
        return path.absolute()
        
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


def get_unique_filename(
    directory: Union[str, Path],
    base_name: str,
    extension: str
) -> Path:
    """
    Generate unique filename by adding counter if file exists.
    
    Parameters:
    -----------
    directory : str or Path
        Directory for file
    base_name : str
        Base filename (without extension)
    extension : str
        File extension (with or without dot)
    
    Returns:
    --------
    Path : Unique file path
    
    Example:
    --------
    >>> get_unique_filename('/tmp', 'result', '.png')
    Path('/tmp/result.png')  # or result_1.png if exists
    """
    directory = Path(directory)
    
    # Ensure extension starts with dot
    if not extension.startswith('.'):
        extension = '.' + extension
    
    # Try base name first
    file_path = directory / f"{base_name}{extension}"
    
    if not file_path.exists():
        return file_path
    
    # Add counter
    counter = 1
    while True:
        file_path = directory / f"{base_name}_{counter}{extension}"
        if not file_path.exists():
            return file_path
        counter += 1
        
        if counter > 10000:
            raise RuntimeError("Too many existing files, cannot generate unique name")


def create_thumbnail(
    img: np.ndarray,
    max_size: Tuple[int, int] = (256, 256),
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Create thumbnail of image.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    max_size : tuple, default=(256, 256)
        Maximum (width, height)
    maintain_aspect : bool, default=True
        Maintain aspect ratio
    
    Returns:
    --------
    numpy.ndarray : Thumbnail image
    """
    h, w = img.shape[:2]
    max_w, max_h = max_size
    
    if maintain_aspect:
        # Calculate scale to fit within max_size
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w, new_h = max_w, max_h
    
    thumbnail = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return thumbnail


def visualize_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create visualization of mask overlaid on image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        RGB image (H, W, 3)
    mask : numpy.ndarray
        Binary mask (H, W) with values 0 or 1
    color : tuple, default=(0, 255, 0)
        RGB color for mask
    alpha : float, default=0.5
        Transparency (0=transparent, 1=opaque)
    
    Returns:
    --------
    numpy.ndarray : Overlay visualization
    """
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    overlay = image.copy()
    
    # Convert mask to binary
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Create colored overlay
    color_mask = np.zeros_like(image)
    color_mask[:, :] = color
    
    # Blend where mask is active
    overlay[binary_mask == 1] = cv2.addWeighted(
        image[binary_mask == 1],
        1 - alpha,
        color_mask[binary_mask == 1],
        alpha,
        0
    )
    
    return overlay


def create_comparison_image(
    images: list,
    titles: list = None,
    grid_cols: int = 2
) -> np.ndarray:
    """
    Create grid comparison of multiple images.
    
    Parameters:
    -----------
    images : list
        List of numpy arrays (images)
    titles : list, optional
        List of titles for each image
    grid_cols : int, default=2
        Number of columns in grid
    
    Returns:
    --------
    numpy.ndarray : Combined comparison image
    """
    if not images:
        raise ValueError("No images provided")
    
    # Ensure all images same size
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    
    # Resize all to same size
    resized = []
    for img in images:
        if img.shape[:2] != (max_h, max_w):
            img_resized = cv2.resize(img, (max_w, max_h))
        else:
            img_resized = img.copy()
        
        # Convert grayscale to RGB
        if img_resized.ndim == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
        resized.append(img_resized)
    
    # Calculate grid dimensions
    grid_rows = (len(resized) + grid_cols - 1) // grid_cols
    
    # Create grid
    rows = []
    for i in range(grid_rows):
        row_images = resized[i * grid_cols:(i + 1) * grid_cols]
        
        # Pad with black images if needed
        while len(row_images) < grid_cols:
            row_images.append(np.zeros_like(resized[0]))
        
        row = np.hstack(row_images)
        rows.append(row)
    
    comparison = np.vstack(rows)
    
    return comparison


class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, step: int = None, message: str = None):
        """Update progress."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        percentage = (self.current_step / self.total_steps) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.current_step > 0:
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
        else:
            eta = 0
        
        msg = f"{self.description}: {self.current_step}/{self.total_steps} ({percentage:.1f}%)"
        if message:
            msg += f" - {message}"
        msg += f" [ETA: {eta:.1f}s]"
        
        logger.info(msg)
    
    def finish(self):
        """Mark as complete."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.description}: Complete in {elapsed:.2f}s")


def format_bytes(size_bytes: int) -> str:
    """
    Format bytes into human-readable string.
    
    Parameters:
    -----------
    size_bytes : int
        Size in bytes
    
    Returns:
    --------
    str : Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def validate_image_array(img: np.ndarray, name: str = "image") -> None:
    """
    Validate image array and raise descriptive error if invalid.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Image to validate
    name : str
        Name for error messages
    
    Raises:
    -------
    ValueError : If image is invalid
    """
    if img is None:
        raise ValueError(f"{name} is None")
    
    if not isinstance(img, np.ndarray):
        raise ValueError(f"{name} must be numpy array, got {type(img)}")
    
    if img.size == 0:
        raise ValueError(f"{name} is empty")
    
    if img.ndim not in [2, 3]:
        raise ValueError(f"{name} must be 2D or 3D array, got {img.ndim}D")
    
    if img.ndim == 3 and img.shape[2] not in [1, 3, 4]:
        raise ValueError(f"{name} has invalid number of channels: {img.shape[2]}")


# Convenience function for common save operation
def quick_save(img: np.ndarray, filename: str, output_dir: str = "output") -> str:
    """
    Quick save with automatic directory creation.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Image to save
    filename : str
        Filename (with extension)
    output_dir : str, default="output"
        Output directory
    
    Returns:
    --------
    str : Path to saved file
    """
    output_path = Path(output_dir) / filename
    return save_image(img, output_path, create_dirs=True)


if __name__ == "__main__":
    # Self-test
    print("=" * 60)
    print("Utils Module Self-Test")
    print("=" * 60)
    
    # Test image saving
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    try:
        path = quick_save(test_img, "test.png", "temp")
        print(f"✓ Image saved: {path}")
        
        info = get_file_info(path)
        print(f"✓ File info: {info['size_bytes']} bytes")
        
        # Clean up
        os.remove(path)
        print("✓ Test file removed")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    print("\n" + "=" * 60)