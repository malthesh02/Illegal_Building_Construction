"""
Enhanced Single Image Pipeline with GPS Coordinate Support
SAFE VERSION (No KeyError, robust legality checks)
"""

import os
import uuid
import numpy as np
import cv2
import logging
import time

from config import SEG_MODEL_DIR, MAX_IMAGE_DIM, DATABASE_PATH
from cv_module import load_image_rgb, resize_keep_aspect
from detector import IllegalBuildingDetector
from utils import save_image
from permit_manager import PermitManager

logger = logging.getLogger(__name__)


def extract_building_metadata(seg_mask, image, permit_manager, img_width, img_height):
    """
    Extract building metadata WITH SAFE GPS COORDINATE VALIDATION
    """
    buildings = []
    
    # Convert to binary mask
    if seg_mask.dtype != np.uint8:
        building_mask = (seg_mask > 0).astype(np.uint8) * 255
    else:
        building_mask = seg_mask.copy()
    
    logger.info(f"Building mask - Non-zero pixels: {np.sum(building_mask > 0)}")
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"Found {len(contours)} contours")
    
    h, w = image.shape[:2]
    filtered_count = 0
    min_area = 50
    
    legal_count = 0
    illegal_count = 0
    uncertain_count = 0

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            filtered_count += 1
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + bw // 2, y + bh // 2

        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            shape_score = min(1.0, circularity * 2)
        else:
            circularity = 0
            shape_score = 0.5

        size_score = min(1.0, area / 5000)
        confidence = min(95, max(30, 50 + (shape_score * 25) + (size_score * 25)))

        building = {
            'id': f'BLDG_{uuid.uuid4().hex[:8]}',
            'area_pixels': int(area),
            'center_x': int(cx),
            'center_y': int(cy),
            'bbox': {'x': int(x), 'y': int(y), 'width': int(bw), 'height': int(bh)},
            'confidence': float(round(confidence, 2)),
            'perimeter': float(perimeter),
            'circularity': float(round(circularity, 3))
        }

        # ✅ Safe legality + GPS check
        try:
            legality = permit_manager.check_building_legality(building, img_width, img_height)
        except Exception as e:
            logger.error(f"Error checking legality for building {building['id']}: {e}")
            legality = {
                'status': 'UNCERTAIN',
                'reason': 'Legality check failed',
                'confidence': 0.0,
                'gps_coordinates': {'lat': None, 'lon': None}
            }

        gps_info = legality.get('gps_coordinates', {})
        lat = gps_info.get('lat', None)
        lon = gps_info.get('lon', None)

# FIX: Ensure GPS coordinates are always numbers, never null
    if lat is None or lon is None:
    # Set default coordinates (center of default bounds)
        lat = (permit_manager.image_bounds['north'] + permit_manager.image_bounds['south']) / 2
        lon = (permit_manager.image_bounds['east'] + permit_manager.image_bounds['west']) / 2
        logger.warning(f"Missing GPS coordinates for building {building['id']}, using default: ({lat}, {lon})")

        building['gps_latitude'] = float(lat)  # Ensure it's a float, not None
        building['gps_longitude'] = float(lon)
        building['area_m2'] = building.get('estimated_area_m2', int(area * 0.25))
        building['legality'] = legality
        building['is_legal'] = legality.get('status', 'UNCERTAIN') == 'LEGAL'
        building['legal_status'] = legality.get('status', 'UNCERTAIN')
        building['legal_reason'] = legality.get('reason', 'Unknown')
        building['legal_confidence'] = legality.get('confidence', 0.0)

        if building['legal_status'] == 'LEGAL':
            legal_count += 1
        elif building['legal_status'] == 'ILLEGAL':
            illegal_count += 1
        else:
            uncertain_count += 1

        buildings.append(building)
        logger.info(
            f"Building {building['id']}: "
            f"GPS=({lat if lat else 'None'}, {lon if lon else 'None'}), "
            f"area={area}px (~{building['area_m2']}m²), "
            f"confidence={confidence:.1f}%, status={building['legal_status']}"
        )

    logger.info(f"Filtered out {filtered_count} small contours (< {min_area} pixels)")
    logger.info(f"Total buildings detected: {len(buildings)}")
    logger.info(f"Legal: {legal_count}, Illegal: {illegal_count}, Uncertain: {uncertain_count}")

    return buildings


def run_single_image_pipeline(image_path, uid=None, results_dir=None, image_bounds=None):
    """
    Process single image with TRUE GPS-based illegal building detection
    """
    start_time = time.time()
    uid = uid or uuid.uuid4().hex
    results_dir = results_dir or os.path.join(os.getcwd(), "data", "results")
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Starting GPS-enabled illegal building detection for session {uid}")
    logger.info(f"Input image: {image_path}")

    permit_manager = PermitManager(DATABASE_PATH)

    if image_bounds:
        permit_manager.set_image_bounds(
            image_bounds['north'],
            image_bounds['south'],
            image_bounds['east'],
            image_bounds['west']
        )
        logger.info(f"Using custom image bounds: {image_bounds}")
    else:
        logger.info(f"Using default bounds: {permit_manager.image_bounds}")

    img = load_image_rgb(image_path)
    logger.info(f"Original size: {img.shape}")

    img = resize_keep_aspect(img, MAX_IMAGE_DIM)
    h, w = img.shape[:2]
    logger.info(f"Resized to: {h}x{w}")

    detector = IllegalBuildingDetector(model_dir=SEG_MODEL_DIR)
    seg_mask = detector.segment_image(img)
    building_pixels = np.sum(seg_mask > 0)
    building_percent = (building_pixels / (h * w)) * 100
    logger.info(f"Segmentation - Building pixels: {building_pixels} ({building_percent:.2f}%)")

    logger.info("Extracting buildings and checking legality with GPS...")
    buildings_metadata = extract_building_metadata(seg_mask, img, permit_manager, w, h)

    legal_buildings = [b for b in buildings_metadata if b['legal_status'] == 'LEGAL']
    illegal_buildings = [b for b in buildings_metadata if b['legal_status'] == 'ILLEGAL']
    uncertain_buildings = [b for b in buildings_metadata if b['legal_status'] == 'UNCERTAIN']

    logger.info(f"Classification - Legal: {len(legal_buildings)}, "
                f"Illegal: {len(illegal_buildings)}, Uncertain: {len(uncertain_buildings)}")

    out_original = os.path.join(results_dir, f"original_{uid}.png")
    save_image(img, out_original)

    overlay_all = overlay_mask_on_image(img, seg_mask * 255, color=(0, 255, 0), alpha=0.5)
    out_segmentation = os.path.join(results_dir, f"segmentation_{uid}.png")
    save_image(overlay_all, out_segmentation)

    overlay_labeled = create_legal_status_overlay(img, buildings_metadata)
    out_detection = os.path.join(results_dir, f"detection_{uid}.png")
    save_image(overlay_labeled, out_detection)

    if illegal_buildings:
        illegal_overlay = create_illegal_buildings_overlay(img, illegal_buildings)
        out_illegal = os.path.join(results_dir, f"illegal_only_{uid}.png")
        save_image(illegal_overlay, out_illegal)
    else:
        out_illegal = None
        logger.info("No illegal buildings detected")

    debug_dir = os.path.join(results_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, f"seg_mask_{uid}.png"), seg_mask * 255)

    stats = {
        'total_pixels': int(h * w),
        'building_pixels': int(building_pixels),
        'building_percentage': float(round(building_percent, 2)),
        'num_buildings_detected': len(buildings_metadata),
        'num_legal_buildings': len(legal_buildings),
        'num_illegal_buildings': len(illegal_buildings),
        'num_uncertain_buildings': len(uncertain_buildings),
        'total_building_area_m2': int(sum(b['area_m2'] for b in buildings_metadata)),
        'illegal_building_area_m2': int(sum(b['area_m2'] for b in illegal_buildings)),
        'average_building_size': int(np.mean([b['area_pixels'] for b in buildings_metadata])) if buildings_metadata else 0,
        'largest_building': int(max([b['area_pixels'] for b in buildings_metadata])) if buildings_metadata else 0,
        'smallest_building': int(min([b['area_pixels'] for b in buildings_metadata])) if buildings_metadata else 0,
        'gps_enabled': True,
        'coordinate_system': 'GPS',
        'image_bounds': permit_manager.image_bounds,
        'compliance_rate': round((len(legal_buildings) / len(buildings_metadata) * 100), 2) if buildings_metadata else 0
    }

    processing_time = time.time() - start_time
    logger.info(f"Pipeline completed in {processing_time:.2f} seconds")

    geojson_path = os.path.join(results_dir, f"permits_{uid}.geojson")
    permit_manager.export_permits_to_geojson(geojson_path)

    result = {
        "original": out_original,
        "segmentation": out_segmentation,
        "detection_overlay": out_detection,
        "illegal_only_overlay": out_illegal,
        "buildings": buildings_metadata,
        "legal_buildings": legal_buildings,
        "illegal_buildings": illegal_buildings,
        "uncertain_buildings": uncertain_buildings,
        "statistics": stats,
        "image_dimensions": {"width": w, "height": h},
        "processing_time": f"{processing_time:.2f}s",
        "legality_summary": {
            'total': len(buildings_metadata),
            'legal': len(legal_buildings),
            'illegal': len(illegal_buildings),
            'uncertain': len(uncertain_buildings),
            'compliance_rate': stats['compliance_rate']
        },
        "geojson_export": geojson_path,
        "gps_enabled": True
    }

    return result


def create_legal_status_overlay(image, buildings):
    """
    Create overlay with GPS-aware legal status visualization
    Green = Legal, Red = Illegal, Yellow = Uncertain
    (Safe version: handles None GPS)
    """
    overlay = image.copy()
    
    for idx, building in enumerate(buildings):
        cx, cy = building['center_x'], building['center_y']
        bbox = building['bbox']
        status = building.get('legal_status', 'UNCERTAIN')
        
        # Safely get GPS coordinates
        lat = building.get('gps_latitude')
        lon = building.get('gps_longitude')
        if lat is None or lon is None:
            gps_label = "GPS: N/A"
        else:
            gps_label = f"GPS: {lat:.4f}, {lon:.4f}"
        
        # Color by status
        if status == 'LEGAL':
            color = (0, 255, 0)
            box_color = (0, 200, 0)
        elif status == 'ILLEGAL':
            color = (255, 0, 0)
            box_color = (200, 0, 0)
        else:
            color = (255, 255, 0)
            box_color = (200, 200, 0)
        
        # Draw bounding box and label
        cv2.rectangle(overlay,
                     (bbox['x'], bbox['y']),
                     (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                     box_color, 2)
        
        cv2.circle(overlay, (cx, cy), 5, color, -1)
        cv2.circle(overlay, (cx, cy), 7, (0, 0, 0), 2)
        
        label = f"#{idx+1} {status}"
        
        cv2.putText(overlay, label, (cx + 12, cy - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(overlay, gps_label, (cx + 12, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    return overlay


def create_illegal_buildings_overlay(image, illegal_buildings):
    """
    Create overlay showing ONLY illegal buildings with GPS coordinates (Safe)
    """
    overlay = image.copy()
    red_layer = np.zeros_like(image)
    red_layer[:, :] = (255, 0, 0)
    
    for idx, building in enumerate(illegal_buildings):
        cx, cy = building['center_x'], building['center_y']
        bbox = building['bbox']
        
        lat = building.get('gps_latitude')
        lon = building.get('gps_longitude')
        if lat is None or lon is None:
            gps_text = "GPS: N/A"
        else:
            gps_text = f"{lat:.4f}, {lon:.4f}"
        
        y1, y2 = bbox['y'], bbox['y'] + bbox['height']
        x1, x2 = bbox['x'], bbox['x'] + bbox['width']
        y1, y2 = max(0, y1), min(image.shape[0], y2)
        x1, x2 = max(0, x1), min(image.shape[1], x2)
        
        if y2 > y1 and x2 > x1:
            roi = overlay[y1:y2, x1:x2]
            red_roi = red_layer[y1:y2, x1:x2]
            overlay[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.6, red_roi, 0.4, 0)
        
        cv2.rectangle(overlay,
                     (bbox['x'], bbox['y']),
                     (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                     (255, 0, 0), 3)
        
        cv2.line(overlay, (cx - 10, cy - 10), (cx + 10, cy + 10), (255, 255, 255), 3)
        cv2.line(overlay, (cx - 10, cy + 10), (cx + 10, cy - 10), (255, 255, 255), 3)
        
        label = f"ILLEGAL #{idx+1}"
        reason = building.get('legal_reason', 'Unknown')[:35]
        
        cv2.putText(overlay, label, (cx - 55, cy - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(overlay, gps_text, (cx - 55, cy - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(overlay, reason, (cx - 55, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return overlay


def overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.5):
    if mask.ndim == 3:
        mask_chan = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_chan = mask
    mask_bin = (mask_chan > 0).astype(np.uint8)
    overlay = image.copy()
    color_layer = np.zeros_like(image, dtype=np.uint8)
    color_layer[:, :] = color
    mask_3d = np.stack([mask_bin] * 3, axis=-1)
    overlay = np.where(mask_3d > 0,
                      cv2.addWeighted(image, 1 - alpha, color_layer, alpha, 0),
                      image)
    return overlay.astype(np.uint8)
