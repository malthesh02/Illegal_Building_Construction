

import numpy as np
import cv2
from sklearn.cluster import KMeans
import warnings

# Suppress sklearn convergence warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def preprocess_image(img):
    """
    Preprocess image for better change detection.
    Applies CLAHE for contrast enhancement.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced

def kmeans_change_detection(img1, img2, k=2, downsample=2, preprocess=True):
    """
    Detect changes between two images using K-means clustering.
    
    Parameters:
    -----------
    img1, img2 : numpy.ndarray
        RGB uint8 images of same shape (H, W, 3)
    k : int, default=2
        Number of clusters (2 = changed vs unchanged)
    downsample : int, default=2
        Downsampling factor for clustering speed (1 = no downsampling)
    preprocess : bool, default=True
        Apply CLAHE preprocessing for better contrast
    
    Returns:
    --------
    mask : numpy.ndarray
        Binary mask (uint8) with 255 for changed pixels, 0 for unchanged
    
    Raises:
    -------
    AssertionError : If images have different shapes
    """
    # Input validation
    assert img1.shape == img2.shape, f"Images must have same shape. Got {img1.shape} and {img2.shape}"
    assert img1.dtype == np.uint8 and img2.dtype == np.uint8, "Images must be uint8"
    assert k >= 2, "k must be at least 2 for change detection"
    
    h, w = img1.shape[:2]
    
    # Optional preprocessing
    if preprocess:
        img1 = preprocess_image(img1)
        img2 = preprocess_image(img2)
    
    # Compute absolute difference
    diff = cv2.absdiff(img1, img2)
    
    # Calculate magnitude of change (Euclidean distance in RGB space)
    mag = np.sqrt((diff.astype(np.float32) ** 2).sum(axis=2))
    
    # Check if images are identical (no changes)
    if mag.max() < 1e-6:
        print("Warning: Images appear identical. No changes detected.")
        return np.zeros((h, w), dtype=np.uint8)
    
    # Normalize magnitude to [0, 255] for better clustering
    mag_normalized = ((mag - mag.min()) / (mag.max() - mag.min()) * 255).astype(np.float32)
    
    # Downsample for faster clustering if requested
    sample = mag_normalized
    if downsample > 1:
        sample = mag_normalized[::downsample, ::downsample]
    
    # Reshape for K-means (needs N x 1 array)
    X = sample.reshape(-1, 1)
    
    # K-means clustering with improved parameters
    # n_init=10 ensures stable clustering (default in older sklearn)
    # max_iter=300 allows more iterations for convergence
    kmeans = KMeans(
        n_clusters=k, 
        random_state=42,  # For reproducibility
        n_init=10,        # FIXED: Was 4, now 10 for better convergence
        max_iter=300,
        algorithm='lloyd'  # Explicit algorithm selection
    )
    
    kmeans.fit(X)
    
    # Predict labels for full image
    labels = kmeans.predict(mag_normalized.reshape(-1, 1)).reshape(h, w)
    
    # Identify which cluster represents change
    # Method: cluster with higher mean magnitude is the "changed" cluster
    cluster_means = []
    for i in range(k):
        cluster_pixels = mag_normalized.reshape(-1)[labels.reshape(-1) == i]
        if len(cluster_pixels) > 0:
            cluster_means.append(cluster_pixels.mean())
        else:
            cluster_means.append(0)
    
    # FIXED: More robust change cluster identification
    change_cluster = int(np.argmax(cluster_means))
    
    # Create binary mask
    mask = (labels == change_cluster).astype(np.uint8) * 255
    
    # Post-processing: Clean up noise with morphological operations
    # FIXED: Adaptive kernel size based on image dimensions
    kernel_size = max(3, min(7, int(min(h, w) / 200)))
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Remove small noise (opening)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill small holes (closing)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Additional filtering: Remove very small connected components
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Calculate minimum area threshold (0.01% of image area)
    min_area = (h * w) * 0.0001
    
    # Filter out small components
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_mask[labels_im == i] = 255
    
    return filtered_mask

def multi_scale_change_detection(img1, img2, scales=[1.0, 0.5]):
    """
    Advanced: Multi-scale change detection for better results.
    Detects changes at multiple resolutions and combines them.
    
    Parameters:
    -----------
    img1, img2 : numpy.ndarray
        RGB uint8 images
    scales : list of float
        Scaling factors to use (1.0 = original size)
    
    Returns:
    --------
    mask : numpy.ndarray
        Combined binary mask
    """
    h, w = img1.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.float32)
    
    for scale in scales:
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scaled_img2 = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scaled_img1, scaled_img2 = img1, img2
        
        # Detect changes at this scale
        mask = kmeans_change_detection(scaled_img1, scaled_img2, k=2, downsample=2)
        
        # Resize back to original size
        if scale != 1.0:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Accumulate
        combined_mask += mask.astype(np.float32) / 255.0
    
    # Normalize and threshold
    combined_mask = combined_mask / len(scales)
    final_mask = (combined_mask > 0.5).astype(np.uint8) * 255
    
    return final_mask

def get_change_statistics(mask):
    """
    Calculate statistics about detected changes.
    """
    total_pixels = mask.size
    changed_pixels = np.sum(mask > 0)
    change_percentage = (changed_pixels / total_pixels) * 100

    # Find connected components (separate changed regions)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Real number of changed regions (exclude background)
    num_regions = num_labels - 1

    # Calculate areas of each region
    region_areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        region_areas.append(area)

    return {
        'total_pixels': total_pixels,
        'changed_pixels': int(changed_pixels),
        'change_percentage': float(change_percentage),
        'num_regions': int(num_regions),
        'region_areas': region_areas,
        'largest_region_area': int(max(region_areas)) if region_areas else 0,
        'mean_region_area': float(np.mean(region_areas)) if region_areas else 0
    }
