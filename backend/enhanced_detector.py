
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box with confidence and metadata"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    area: int
    label: str = "building"
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def corners(self) -> List[Tuple[int, int]]:
        """Get corner points (clockwise from top-left)"""
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height)
        ]
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another box"""
        # Intersection coordinates
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        # No intersection
        if x2 < x1 or y2 < y1:
            return 0.0
        
        # Calculate areas
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'confidence': round(self.confidence, 3),
            'area': self.area,
            'center': self.center,
            'label': self.label
        }


class RegionProposalDetector:
    """
    Building detector using region proposals and smart filtering
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize detector
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        
        # Detection parameters
        self.min_area = self.config.get('min_area', 50)
        self.max_area_ratio = self.config.get('max_area_ratio', 0.5)
        self.min_aspect_ratio = self.config.get('min_aspect_ratio', 0.2)
        self.max_aspect_ratio = self.config.get('max_aspect_ratio', 5.0)
        self.nms_threshold = self.config.get('nms_threshold', 0.3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        # Multi-scale parameters
        self.scales = self.config.get('scales', [1.0, 0.8, 1.2])
        
        logger.info(f"RegionProposalDetector initialized with:")
        logger.info(f"  Min area: {self.min_area} px")
        logger.info(f"  NMS threshold: {self.nms_threshold}")
        logger.info(f"  Scales: {self.scales}")
    
    def detect(self, image: np.ndarray, mask: np.ndarray) -> List[BoundingBox]:
        """
        Detect buildings using region proposals
        
        Args:
            image: RGB image (H, W, 3)
            mask: Binary segmentation mask (H, W)
        
        Returns:
            List of BoundingBox objects
        """
        h, w = image.shape[:2]
        logger.info(f"Starting detection on {w}x{h} image")
        
        # Generate region proposals
        proposals = self._generate_proposals(mask, (h, w))
        logger.info(f"Generated {len(proposals)} initial proposals")
        
        if not proposals:
            logger.warning("No proposals generated")
            return []
        
        # Filter by geometry
        filtered_proposals = self._filter_proposals(proposals, (h, w))
        logger.info(f"Filtered to {len(filtered_proposals)} valid proposals")
        
        if not filtered_proposals:
            return []
        
        # Score proposals
        scored_proposals = self._score_proposals(filtered_proposals, image, mask)
        logger.info(f"Scored {len(scored_proposals)} proposals")
        
        # Apply confidence threshold
        confident_proposals = [p for p in scored_proposals if p.confidence >= self.confidence_threshold]
        logger.info(f"Kept {len(confident_proposals)} proposals above confidence threshold")
        
        if not confident_proposals:
            return []
        
        # Non-Maximum Suppression
        final_proposals = self._apply_nms(confident_proposals)
        logger.info(f"Final detection: {len(final_proposals)} buildings after NMS")
        
        return final_proposals
    
    def _generate_proposals(self, mask: np.ndarray, image_size: Tuple[int, int]) -> List[BoundingBox]:
        """
        Generate region proposals using multiple methods
        
        Args:
            mask: Binary mask
            image_size: (height, width)
        
        Returns:
            List of proposal bounding boxes
        """
        proposals = []
        h, w = image_size
        
        # Method 1: Connected Components
        proposals.extend(self._proposals_from_connected_components(mask))
        
        # Method 2: Contour-based
        proposals.extend(self._proposals_from_contours(mask))
        
        # Method 3: Sliding window (for missed regions)
        proposals.extend(self._proposals_from_sliding_window(mask, image_size))
        
        return proposals
    
    def _proposals_from_connected_components(self, mask: np.ndarray) -> List[BoundingBox]:
        """Generate proposals from connected components"""
        proposals = []
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        
        for i in range(1, num_labels):  # Skip background (0)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= self.min_area:
                proposals.append(BoundingBox(
                    x=int(x),
                    y=int(y),
                    width=int(width),
                    height=int(height),
                    confidence=0.5,  # Initial confidence
                    area=int(area)
                ))
        
        return proposals
    
    def _proposals_from_contours(self, mask: np.ndarray) -> List[BoundingBox]:
        """Generate proposals from contours with rotated rectangles"""
        proposals = []
        
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area:
                continue
            
            # Standard bounding box
            x, y, width, height = cv2.boundingRect(contour)
            proposals.append(BoundingBox(
                x=int(x),
                y=int(y),
                width=int(width),
                height=int(height),
                confidence=0.6,
                area=int(area)
            ))
            
            # Rotated rectangle (for oriented buildings)
            if len(contour) >= 5:
                try:
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # Convert rotated rect to axis-aligned bbox
                    x_coords = box[:, 0]
                    y_coords = box[:, 1]
                    
                    x_min, x_max = x_coords.min(), x_coords.max()
                    y_min, y_max = y_coords.min(), y_coords.max()
                    
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    if width > 0 and height > 0:
                        proposals.append(BoundingBox(
                            x=int(x_min),
                            y=int(y_min),
                            width=int(width),
                            height=int(height),
                            confidence=0.7,
                            area=int(area)
                        ))
                except:
                    pass
        
        return proposals
    
    def _proposals_from_sliding_window(self, mask: np.ndarray, 
                                       image_size: Tuple[int, int]) -> List[BoundingBox]:
        """
        Generate proposals using sliding window (catches missed regions)
        
        Args:
            mask: Binary mask
            image_size: (height, width)
        
        Returns:
            List of proposals
        """
        proposals = []
        h, w = image_size
        
        # Define window sizes (multiple scales)
        window_sizes = [
            (32, 32), (64, 64), (96, 96), (128, 128),
            (48, 96), (96, 48),  # Rectangular windows
        ]
        
        stride = 32  # Step size
        
        for win_h, win_w in window_sizes:
            for y in range(0, h - win_h + 1, stride):
                for x in range(0, w - win_w + 1, stride):
                    # Extract window from mask
                    window = mask[y:y+win_h, x:x+win_w]
                    
                    # Check if window contains enough building pixels
                    building_pixels = np.sum(window > 0)
                    window_area = win_h * win_w
                    
                    # At least 30% building coverage
                    if building_pixels / window_area > 0.3:
                        proposals.append(BoundingBox(
                            x=int(x),
                            y=int(y),
                            width=int(win_w),
                            height=int(win_h),
                            confidence=0.4,
                            area=int(building_pixels)
                        ))
        
        return proposals
    
    def _filter_proposals(self, proposals: List[BoundingBox], 
                         image_size: Tuple[int, int]) -> List[BoundingBox]:
        """
        Filter proposals based on geometric constraints
        
        Args:
            proposals: List of proposals
            image_size: (height, width)
        
        Returns:
            Filtered proposals
        """
        filtered = []
        h, w = image_size
        max_area = h * w * self.max_area_ratio
        
        for proposal in proposals:
            # Check area
            if proposal.area < self.min_area or proposal.area > max_area:
                continue
            
            # Check aspect ratio
            if proposal.width == 0 or proposal.height == 0:
                continue
            
            aspect_ratio = proposal.width / proposal.height
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Check bounds
            if proposal.x < 0 or proposal.y < 0:
                continue
            if proposal.x + proposal.width > w or proposal.y + proposal.height > h:
                continue
            
            # Check minimum dimensions
            if proposal.width < 10 or proposal.height < 10:
                continue
            
            filtered.append(proposal)
        
        return filtered
    
    def _score_proposals(self, proposals: List[BoundingBox], 
                        image: np.ndarray, mask: np.ndarray) -> List[BoundingBox]:
        """
        Score proposals based on multiple features
        
        Args:
            proposals: List of proposals
            image: Original image
            mask: Segmentation mask
        
        Returns:
            Proposals with updated confidence scores
        """
        scored = []
        
        for proposal in proposals:
            # Extract region
            x, y, w, h = proposal.x, proposal.y, proposal.width, proposal.height
            
            # Region from mask
            mask_region = mask[y:y+h, x:x+w]
            
            if mask_region.size == 0:
                continue
            
            # Feature 1: Coverage score (how much of bbox is building)
            coverage = np.sum(mask_region > 0) / (w * h)
            
            # Feature 2: Compactness (how square-like)
            aspect_ratio = w / h if h > 0 else 0
            compactness = min(aspect_ratio, 1/aspect_ratio) if aspect_ratio > 0 else 0
            
            # Feature 3: Edge strength
            image_region = image[y:y+h, x:x+w]
            gray_region = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY) if image_region.ndim == 3 else image_region
            edges = cv2.Canny(gray_region, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            
            # Feature 4: Texture variance (buildings have structured texture)
            texture_var = np.var(gray_region) / 255.0
            
            # Weighted combination
            confidence = (
                coverage * 0.4 +
                compactness * 0.2 +
                edge_density * 0.2 +
                texture_var * 0.2
            )
            
            # Combine with initial confidence
            final_confidence = (proposal.confidence + confidence) / 2
            
            # Create new proposal with updated confidence
            scored.append(BoundingBox(
                x=proposal.x,
                y=proposal.y,
                width=proposal.width,
                height=proposal.height,
                confidence=final_confidence,
                area=proposal.area,
                label=proposal.label
            ))
        
        return scored
    
    def _apply_nms(self, proposals: List[BoundingBox]) -> List[BoundingBox]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            proposals: List of proposals with confidence scores
        
        Returns:
            Filtered proposals after NMS
        """
        if not proposals:
            return []
        
        # Sort by confidence (descending)
        proposals = sorted(proposals, key=lambda p: p.confidence, reverse=True)
        
        keep = []
        
        while proposals:
            # Take the highest confidence proposal
            current = proposals.pop(0)
            keep.append(current)
            
            # Remove proposals with high IoU
            proposals = [
                p for p in proposals 
                if current.iou(p) < self.nms_threshold
            ]
        
        return keep
    
    def refine_bounding_boxes(self, proposals: List[BoundingBox], 
                             mask: np.ndarray) -> List[BoundingBox]:
        """
        Refine bounding boxes to fit tightly around building contours
        
        Args:
            proposals: Initial proposals
            mask: Segmentation mask
        
        Returns:
            Refined proposals
        """
        refined = []
        
        for proposal in proposals:
            # Extract mask region
            x, y, w, h = proposal.x, proposal.y, proposal.width, proposal.height
            mask_region = mask[y:y+h, x:x+w]
            
            if mask_region.size == 0:
                continue
            
            # Find non-zero pixels
            coords = np.column_stack(np.where(mask_region > 0))
            
            if len(coords) == 0:
                continue
            
            # Calculate tight bounding box
            y_coords, x_coords = coords[:, 0], coords[:, 1]
            
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            # Convert to global coordinates
            refined_x = x + x_min
            refined_y = y + y_min
            refined_w = x_max - x_min + 1
            refined_h = y_max - y_min + 1
            
            # Calculate refined area
            refined_area = np.sum(mask_region[y_min:y_max+1, x_min:x_max+1] > 0)
            
            refined.append(BoundingBox(
                x=int(refined_x),
                y=int(refined_y),
                width=int(refined_w),
                height=int(refined_h),
                confidence=proposal.confidence,
                area=int(refined_area),
                label=proposal.label
            ))
        
        return refined


class MultiScaleBuildingDetector:
    """
    Multi-scale building detector that combines multiple detection scales
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize multi-scale detector"""
        self.config = config or {}
        self.scales = self.config.get('scales', [0.8, 1.0, 1.2])
        self.base_detector = RegionProposalDetector(config)
        
        logger.info(f"MultiScaleBuildingDetector initialized with scales: {self.scales}")
    
    def detect(self, image: np.ndarray, mask: np.ndarray) -> List[BoundingBox]:
        """
        Detect buildings at multiple scales
        
        Args:
            image: RGB image
            mask: Segmentation mask
        
        Returns:
            Combined detections from all scales
        """
        all_proposals = []
        original_h, original_w = image.shape[:2]
        
        for scale in self.scales:
            logger.info(f"Processing scale: {scale}")
            
            # Resize image and mask
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            
            scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Detect at this scale
            scale_proposals = self.base_detector.detect(scaled_image, scaled_mask)
            
            # Scale bounding boxes back to original size
            for proposal in scale_proposals:
                original_proposal = BoundingBox(
                    x=int(proposal.x / scale),
                    y=int(proposal.y / scale),
                    width=int(proposal.width / scale),
                    height=int(proposal.height / scale),
                    confidence=proposal.confidence,
                    area=int(proposal.area / (scale ** 2)),
                    label=proposal.label
                )
                all_proposals.append(original_proposal)
        
        # Apply NMS across all scales
        final_proposals = self.base_detector._apply_nms(all_proposals)
        
        # Refine final bounding boxes
        refined_proposals = self.base_detector.refine_bounding_boxes(final_proposals, mask)
        
        logger.info(f"Multi-scale detection complete: {len(refined_proposals)} buildings")
        
        return refined_proposals


def visualize_detections(image: np.ndarray, detections: List[BoundingBox], 
                        show_confidence: bool = True) -> np.ndarray:
    """
    Visualize detections on image
    
    Args:
        image: RGB image
        detections: List of detections
        show_confidence: Whether to show confidence scores
    
    Returns:
        Image with visualizations
    """
    vis_image = image.copy()
    
    for idx, detection in enumerate(detections):
        x, y, w, h = detection.x, detection.y, detection.width, detection.height
        conf = detection.confidence
        
        # Color based on confidence
        if conf > 0.8:
            color = (0, 255, 0)  # Green - high confidence
        elif conf > 0.6:
            color = (255, 255, 0)  # Yellow - medium confidence
        else:
            color = (255, 165, 0)  # Orange - low confidence
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        # Draw center point
        cx, cy = detection.center
        cv2.circle(vis_image, (cx, cy), 3, color, -1)
        
        # Add label
        label = f"#{idx+1}"
        if show_confidence:
            label += f" {conf:.2f}"
        
        cv2.putText(vis_image, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis_image


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced Building Detector - Test")
    print("=" * 70)
    
    # Create test image and mask
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_mask = np.zeros((512, 512), dtype=np.uint8)
    
    # Add some "buildings" to mask
    test_mask[100:150, 100:180] = 255
    test_mask[200:280, 200:250] = 255
    test_mask[350:420, 350:450] = 255
    
    # Test single-scale detector
    print("\n1. Testing Single-Scale Detector...")
    config = {
        'min_area': 50,
        'nms_threshold': 0.3,
        'confidence_threshold': 0.4
    }
    
    detector = RegionProposalDetector(config)
    detections = detector.detect(test_image, test_mask)
    
    print(f"✅ Detected {len(detections)} buildings")
    for i, det in enumerate(detections):
        print(f"   Building {i+1}: {det.to_dict()}")
    
    # Test multi-scale detector
    print("\n2. Testing Multi-Scale Detector...")
    multi_detector = MultiScaleBuildingDetector(config)
    multi_detections = multi_detector.detect(test_image, test_mask)
    
    print(f"✅ Multi-scale detected {len(multi_detections)} buildings")
    
    # Visualize
    print("\n3. Creating visualization...")
    vis_result = visualize_detections(test_image, multi_detections)
    print("✅ Visualization created")
    
    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)