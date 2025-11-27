import sys
import os
import uuid
import numpy as np
import tensorflow as tf
import cv2
import logging
from datetime import datetime

# TensorFlow configuration
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Flask imports
from flask import Flask, redirect, url_for, request, render_template, flash, jsonify, send_file
from werkzeug.utils import secure_filename

# Import our detection modules
from config import DATABASE_PATH, UPLOAD_DIR, RESULTS_DIR, SEG_MODEL_DIR, MAX_IMAGE_DIM
from cv_module import load_image_rgb, resize_keep_aspect, match_image_sizes
from detector import IllegalBuildingDetector
from change_detection import kmeans_change_detection, get_change_statistics
from database_manager import EnhancedDatabaseManager
from permit_manager import PermitManager
from report_generator import PDFReportGenerator
from gps_calculator import GPSCoordinateCalculator, enhance_building_with_accurate_gps
from enhanced_detector import RegionProposalDetector, MultiScaleBuildingDetector, BoundingBox, visualize_detections
from cadastral_integration import CadastralMapIntegration
from werkzeug.security import generate_password_hash, check_password_hash
from utils import save_image

# ===== NEW: IMPORT GPS CALCULATOR =====
from gps_calculator import GPSCoordinateCalculator, enhance_building_with_accurate_gps

# ===== NEW: IMPORT ENHANCED DETECTOR =====
from enhanced_detector import (
    RegionProposalDetector,
    MultiScaleBuildingDetector,
    BoundingBox,
    visualize_detections
)

import requests
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# ===== FIX CORS ERROR =====
from flask_cors import CORS
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
app.secret_key = "illegal_building_detection_secret_key_2024"
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['RESULTS_FOLDER'] = RESULTS_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'}

# Initialize components
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "illegal_buildings.db")
db_manager = EnhancedDatabaseManager(DATABASE_PATH)

permit_manager = PermitManager(DATABASE_PATH)
report_generator = PDFReportGenerator(output_dir=os.path.join(RESULTS_DIR, 'reports'))
detector = IllegalBuildingDetector(model_dir=SEG_MODEL_DIR)

from cadastral_integration import CadastralMapIntegration
cadastral_integration = CadastralMapIntegration(DATABASE_PATH)

# Session results storage
session_results = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_public_ip():
    """Get public IP address of user"""
    try:
        response = requests.get('https://httpbin.org/ip', timeout=5)
        response.raise_for_status()
        ip_data = response.json()
        return ip_data['origin']
    except Exception as e:
        logger.warning(f"Error getting public IP: {e}")
        return None


def get_location_by_ip(ip_address):
    """Get GPS location from IP address"""
    try:
        response = requests.get(f'https://ipinfo.io/{ip_address}/json', timeout=5)
        response.raise_for_status()
        data = response.json()
        location = data.get('loc')
        if location:
            latitude, longitude = location.split(',')
            return float(latitude), float(longitude)
        else:
            return None, None
    except Exception as e:
        logger.warning(f"Error getting location data: {e}")
        return None, None


def extract_building_metadata(seg_mask, image, change_mask, permit_manager, img_width, img_height,
                              change_threshold=30.0, gps_calculator=None):
    """
    Extract building metadata with ENHANCED REGION PROPOSALS, BOUNDING BOXES, AND GPS

    Args:
        seg_mask: Building segmentation mask
        image: Original image
        change_mask: K-means change detection mask
        permit_manager: PermitManager instance
        img_width: Image width
        img_height: Image height
        change_threshold: Threshold percentage for construction detection (default 30%)
        gps_calculator: GPSCoordinateCalculator instance for accurate GPS calculation

    Returns:
        List of building dictionaries with metadata
    """
    buildings = []

    # Convert to binary mask
    if seg_mask.dtype != np.uint8:
        building_mask = (seg_mask > 0).astype(np.uint8) * 255
    else:
        building_mask = seg_mask.copy()

    logger.info(f"Building mask - Non-zero pixels: {np.sum(building_mask > 0)}")

    # ===== ENHANCED DETECTION WITH REGION PROPOSALS =====
    logger.info("🎯 Using Enhanced Region Proposal Detector...")

    # Configure detector
    detector_config = {
        'min_area': 50,
        'max_area_ratio': 0.5,
        'min_aspect_ratio': 0.2,
        'max_aspect_ratio': 5.0,
        'nms_threshold': 0.3,
        'confidence_threshold': 0.5,
        'scales': [0.9, 1.0, 1.1]  # Multi-scale detection
    }

    # Create multi-scale detector
    region_detector = MultiScaleBuildingDetector(config=detector_config)

    # Detect buildings with region proposals
    bounding_boxes = region_detector.detect(image, building_mask)

    logger.info(f"✅ Detected {len(bounding_boxes)} buildings with region proposals")

    h, w = image.shape[:2]
    legal_count = 0
    illegal_count = 0
    uncertain_count = 0

    for idx, bbox in enumerate(bounding_boxes):
        # Get bounding box properties
        x, y, bw, bh = bbox.x, bbox.y, bbox.width, bbox.height
        cx, cy = bbox.center
        area = bbox.area
        detection_confidence = bbox.confidence

        # Calculate shape features
        if bw > 0 and bh > 0:
            aspect_ratio = bw / bh
            extent = area / (bw * bh)
        else:
            aspect_ratio = 1.0
            extent = 0.5

        # Calculate perimeter from bounding box
        perimeter = 2 * (bw + bh)

        # Circularity (approximation)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        logger.info(f"Building {idx+1}: Area={area}px, BBox=({x},{y},{bw},{bh}), Confidence={detection_confidence:.2f}")

        # ===== K-MEANS CHANGE DETECTION FOR THIS BUILDING =====
        # Extract building region from change mask
        building_region = change_mask[y:y+bh, x:x+bw]

        if building_region.size > 0:
            changed_pixels = np.sum(building_region > 0)
            total_pixels = building_region.size
            change_percentage = (changed_pixels / total_pixels) * 100
        else:
            change_percentage = 0.0

        logger.info(f"  Change: {change_percentage:.1f}%")

        # Create building dictionary
        building = {
            'id': f'BLDG_{uuid.uuid4().hex[:8]}',
            'area_pixels': int(area),
            'center_x': int(cx),
            'center_y': int(cy),
            'bbox': {'x': int(x), 'y': int(y), 'width': int(bw), 'height': int(bh)},
            'confidence': float(round(detection_confidence, 2)),
            'perimeter': float(perimeter),
            'circularity': float(round(circularity, 3)),
            'aspect_ratio': float(round(aspect_ratio, 3)),
            'extent': float(round(extent, 3)),
            'change_percentage': float(round(change_percentage, 2)),
            'detection_method': 'region_proposal'
        }

        # ===== ENHANCED GPS CALCULATION =====
        if gps_calculator:
            # Use accurate GPS calculator
            building = enhance_building_with_accurate_gps(building, gps_calculator)
            logger.info(
                f"  GPS: ({building['gps_latitude']:.6f}, {building['gps_longitude']:.6f}), "
                f"Area: {building['area_m2']:.2f}m², "
                f"Perimeter: {building.get('perimeter_m', 0):.2f}m, "
                f"Accuracy: ±{building.get('gps_accuracy_estimate_m', 0):.2f}m"
            )
        else:
            # Fallback to simple method
            lat, lon = permit_manager.pixel_to_gps(cx, cy, img_width, img_height)
            building['gps_latitude'] = float(lat)
            building['gps_longitude'] = float(lon)
            building['area_m2'] = int(area * 0.25)  # Rough estimate
            logger.warning(f"  Using fallback GPS calculation")

        # ===== APPLY 30% THRESHOLD (AS PER PDF) =====
        if change_percentage > change_threshold:
            building['status'] = 'UNDER_CONSTRUCTION'
            building['requires_license_check'] = True

            # Check legality using permit manager
            try:
                legality = permit_manager.check_building_legality(building, img_width, img_height)
            except Exception as e:
                logger.error(f"Error checking legality for building {building['id']}: {e}")
                legality = {
                    'status': 'UNCERTAIN',
                    'reason': 'Legality check failed',
                    'confidence': 0.0,
                    'gps_coordinates': {'lat': building.get('gps_latitude'),
                                        'lon': building.get('gps_longitude')}
                }

            building['legal_status'] = legality.get('status', 'UNCERTAIN')
            building['legal_reason'] = legality.get('reason', 'Unknown')
            building['legal_confidence'] = legality.get('confidence', 0.0)

        else:
            # Existing building (no significant change)
            building['status'] = 'EXISTING'
            building['requires_license_check'] = False
            building['legal_status'] = 'NOT_CHECKED'
            building['legal_reason'] = f'No significant change (only {change_percentage:.1f}%)'
            building['legal_confidence'] = 0.0

        # Count by legal status
        if building['legal_status'] == 'LEGAL':
            legal_count += 1
        elif building['legal_status'] == 'ILLEGAL':
            illegal_count += 1
        else:
            uncertain_count += 1

        buildings.append(building)

        logger.info(
            f"  Status: {building['status']}, "
            f"Legal: {building['legal_status']}"
        )

    logger.info(f"📊 Total buildings: {len(buildings)}")
    logger.info(f"  Legal: {legal_count}, Illegal: {illegal_count}, Uncertain: {uncertain_count}")

    return buildings


def create_visualization_overlay(image, buildings, change_mask):
    """
    Create visualization overlay with building markings using enhanced bounding boxes

    Args:
        image: Original image
        buildings: List of detected buildings
        change_mask: K-means change detection mask

    Returns:
        Overlay image with markings
    """
    overlay = image.copy()

    # Draw change mask with transparency
    change_colored = cv2.applyColorMap(change_mask, cv2.COLORMAP_HOT)
    overlay = cv2.addWeighted(overlay, 0.7, change_colored, 0.3, 0)

    for idx, building in enumerate(buildings):
        cx, cy = building['center_x'], building['center_y']
        bbox = building['bbox']
        status = building.get('status', 'UNKNOWN')
        legal_status = building.get('legal_status', 'UNKNOWN')
        change_pct = building.get('change_percentage', 0)
        confidence = building.get('confidence', 0)

        # Determine color based on status
        if status == 'EXISTING':
            color = (0, 255, 0)  # Green - no change
            box_color = (0, 200, 0)
            label = f"E#{idx+1}"
        elif status == 'UNDER_CONSTRUCTION':
            if legal_status == 'LEGAL':
                color = (0, 255, 255)  # Cyan - legal construction
                box_color = (0, 200, 200)
                label = f"L#{idx+1}"
            elif legal_status == 'ILLEGAL':
                color = (0, 0, 255)  # Red - illegal construction
                box_color = (0, 0, 200)
                label = f"I#{idx+1}"
            else:
                color = (255, 255, 0)  # Yellow - uncertain
                box_color = (200, 200, 0)
                label = f"U#{idx+1}"
        else:
            color = (128, 128, 128)  # Gray - unknown
            box_color = (100, 100, 100)
            label = f"?#{idx+1}"

        # Draw bounding box (ENHANCED - thicker for better visibility)
        cv2.rectangle(overlay,
                      (bbox['x'], bbox['y']),
                      (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                      box_color, 3)

        # Draw center point
        cv2.circle(overlay, (cx, cy), 6, color, -1)
        cv2.circle(overlay, (cx, cy), 8, (255, 255, 255), 2)

        # Add labels with confidence
        label_with_conf = f"{label} ({confidence:.0%})"
        cv2.putText(overlay, label_with_conf, (bbox['x'], bbox['y'] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        cv2.putText(overlay, f"Δ{change_pct:.1f}%",
                    (bbox['x'], bbox['y'] + bbox['height'] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

    return overlay


def create_illegal_buildings_overlay(image, illegal_buildings):
    """
    Create overlay showing ONLY illegal buildings with enhanced visualization

    Args:
        image: Original image
        illegal_buildings: List of illegal buildings

    Returns:
        Overlay image with illegal buildings highlighted
    """
    overlay = image.copy()

    for idx, building in enumerate(illegal_buildings):
        cx, cy = building['center_x'], building['center_y']
        bbox = building['bbox']
        change_pct = building.get('change_percentage', 0)
        confidence = building.get('confidence', 0)

        # Highlight in red with thick border
        cv2.rectangle(overlay,
                      (bbox['x'], bbox['y']),
                      (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                      (0, 0, 255), 4)

        # Draw X marker
        cv2.line(overlay, (cx - 15, cy - 15), (cx + 15, cy + 15), (255, 255, 255), 4)
        cv2.line(overlay, (cx - 15, cy + 15), (cx + 15, cy - 15), (255, 255, 255), 4)

        # Labels
        label = f"ILLEGAL #{idx+1}"
        cv2.putText(overlay, label, (bbox['x'], bbox['y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3, cv2.LINE_AA)

        # Add confidence and change info
        info = f"Conf:{confidence:.0%} | Δ{change_pct:.1f}%"
        cv2.putText(overlay,
                    info,
                    (bbox['x'], bbox['y'] + bbox['height'] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return overlay


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    """Upload page for two temporal images"""
    return render_template('upload.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    """
    Main prediction endpoint using:
    - K-MEANS CLUSTERING for change detection
    - ENHANCED REGION PROPOSALS for building detection
    - ACCURATE GPS COORDINATES
    """
    if request.method == 'POST':
        try:
            # Get uploaded files
            file1 = request.files.get('file1')  # Time 1 (older image)
            file2 = request.files.get('file2')  # Time 2 (newer image)

            if not file1 or not file2:
                flash('Please upload both temporal images', 'error')
                return redirect(url_for('upload_page'))

            # Validate file types
            if not allowed_file(file1.filename) or not allowed_file(file2.filename):
                flash('Invalid file type. Please upload images only.', 'error')
                return redirect(url_for('upload_page'))

            # Save uploaded files
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename1 = f"t1_{timestamp}_{filename1}"
            filename2 = f"t2_{timestamp}_{filename2}"

            file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

            file1.save(file1_path)
            file2.save(file2_path)

            logger.info(f"📤 Uploaded files: {file1_path}, {file2_path}")

            # Generate session ID
            session_id = f"session_{uuid.uuid4().hex[:8]}"

            # Get optional GPS bounds (if provided by user)
            image_bounds = None
            try:
                north = request.form.get('north', type=float)
                south = request.form.get('south', type=float)
                east = request.form.get('east', type=float)
                west = request.form.get('west', type=float)

                if all([north, south, east, west]):
                    image_bounds = {
                        'north': north, 'south': south,
                        'east': east, 'west': west
                    }
                    logger.info(f"🗺️  Using GPS bounds: {image_bounds}")
            except:
                logger.info("📍 No GPS bounds provided, using defaults")

            # Get change threshold (default 30% as per PDF)
            change_threshold = float(request.form.get('threshold', 30.0))
            logger.info(f"📊 Using change threshold: {change_threshold}%")

            # ===== STEP 1: LOAD IMAGES =====
            logger.info("=" * 80)
            logger.info(f"🚀 Starting Enhanced Detection for session: {session_id}")
            logger.info("=" * 80)
            logger.info("Step 1: Loading images...")

            img1 = load_image_rgb(file1_path)
            img2 = load_image_rgb(file2_path)

            # Resize images
            img1 = resize_keep_aspect(img1, MAX_IMAGE_DIM)
            img2 = resize_keep_aspect(img2, MAX_IMAGE_DIM)

            # Match sizes
            img1, img2 = match_image_sizes(img1, img2, method='crop')
            h, w = img2.shape[:2]
            logger.info(f"✅ Images prepared: {w}x{h}")

            # ===== STEP 2: K-MEANS CHANGE DETECTION =====
            logger.info("Step 2: Running K-means change detection...")
            change_mask = kmeans_change_detection(
                img1, img2,
                k=2,
                downsample=2,
                preprocess=True
            )

            change_stats = get_change_statistics(change_mask)
            logger.info(f"✅ Change detected: {change_stats['change_percentage']:.2f}%")
            logger.info(f"   Changed pixels: {change_stats['changed_pixels']:,}")
            logger.info(f"   Change regions: {change_stats['num_regions']}")

            # ===== STEP 3: INITIALIZE GPS CALCULATOR =====
            logger.info("Step 3: Initializing GPS calculator...")

            # Set image bounds in permit manager
            if image_bounds:
                permit_manager.set_image_bounds(
                    image_bounds['north'],
                    image_bounds['south'],
                    image_bounds['east'],
                    image_bounds['west']
                )

            # Create GPS calculator
            gps_calculator = GPSCoordinateCalculator(
                image_bounds=image_bounds or permit_manager.image_bounds,
                image_size=(w, h)
            )

            # Log GPS statistics
            gps_stats = gps_calculator.get_statistics()
            logger.info(f"✅ GPS Calculator initialized:")
            logger.info(f"   Resolution: {gps_stats['resolution']['meters_per_pixel_x']:.2f}m/px (X), "
                        f"{gps_stats['resolution']['meters_per_pixel_y']:.2f}m/px (Y)")
            logger.info(f"   Coverage: {gps_stats['coverage_area_km2']:.3f} km²")

            # ===== STEP 4: BUILDING DETECTION WITH SEGMENTATION =====
            logger.info("Step 4: Running building segmentation...")
            seg_mask = detector.segment_image(img2)
            building_pixels = np.sum(seg_mask > 0)
            logger.info(f"✅ Building pixels detected: {building_pixels:,}")

            # ===== STEP 5: EXTRACT BUILDINGS WITH REGION PROPOSALS + GPS =====
            logger.info("Step 5: Extracting buildings with ENHANCED DETECTION...")

            buildings = extract_building_metadata(
                seg_mask,
                img2,
                change_mask,
                permit_manager,
                w, h,
                change_threshold,
                gps_calculator=gps_calculator
            )

            # Categorize buildings
            under_construction = [b for b in buildings if b['status'] == 'UNDER_CONSTRUCTION']
            existing = [b for b in buildings if b['status'] == 'EXISTING']
            illegal = [b for b in buildings if b.get('legal_status') == 'ILLEGAL']
            legal = [b for b in buildings if b.get('legal_status') == 'LEGAL']

            logger.info(f"✅ Buildings categorized:")
            logger.info(f"   Under construction: {len(under_construction)}")
            logger.info(f"   Existing: {len(existing)}")
            logger.info(f"   Illegal: {len(illegal)}")
            logger.info(f"   Legal: {len(legal)}")

            # ===== STEP 6: GENERATE VISUALIZATIONS =====
            logger.info("Step 6: Generating visualizations...")

            # Save original images
            t1_save_path = os.path.join(app.config['RESULTS_FOLDER'], f"t1_{session_id}.png")
            t2_save_path = os.path.join(app.config['RESULTS_FOLDER'], f"t2_{session_id}.png")
            save_image(img1, t1_save_path)
            save_image(img2, t2_save_path)

            # Save change mask
            change_viz = cv2.applyColorMap(change_mask, cv2.COLORMAP_HOT)
            change_path = os.path.join(app.config['RESULTS_FOLDER'], f"change_{session_id}.png")
            save_image(change_viz, change_path)

            # Create detection overlay
            detection_overlay = create_visualization_overlay(img2, buildings, change_mask)
            detection_path = os.path.join(app.config['RESULTS_FOLDER'], f"detection_{session_id}.png")
            save_image(detection_overlay, detection_path)

            # Create illegal buildings overlay
            illegal_path = None
            if illegal:
                illegal_overlay = create_illegal_buildings_overlay(img2, illegal)
                illegal_path = os.path.join(app.config['RESULTS_FOLDER'], f"illegal_{session_id}.png")
                save_image(illegal_overlay, illegal_path)

            logger.info(f"✅ Visualizations saved")

            # ===== STEP 7: CALCULATE STATISTICS =====
            statistics = {
                'total_pixels': int(h * w),
                'change_pixels': int(change_stats['changed_pixels']),
                'change_percentage': float(change_stats['change_percentage']),
                'num_change_regions': int(change_stats['num_regions']),
                'total_buildings': len(buildings),
                'buildings_under_construction': len(under_construction),
                'existing_buildings': len(existing),
                'legal_constructions': len(legal),
                'illegal_constructions': len(illegal),
                'total_building_area_m2': int(sum(b.get('area_m2', 0) for b in buildings)),
                'illegal_building_area_m2': int(sum(b.get('area_m2', 0) for b in illegal)),
                'compliance_rate': round((len(legal) / len(under_construction) * 100), 2) if under_construction else 100.0,
                'change_threshold': change_threshold,
                'gps_resolution_m_per_px': round(gps_stats['resolution']['meters_per_pixel_x'], 3),
                'coverage_area_km2': gps_stats['coverage_area_km2'],
                'detection_method': 'region_proposal + multi_scale'
            }

            # ===== STEP 8: SAVE TO DATABASE =====
            logger.info("Step 7: Saving to database...")
            db_manager.create_session(
                session_id=session_id,
                t1_path=file1_path,
                t2_path=file2_path,
                metadata={
                    'mode': 'two-temporal-kmeans-gps-enhanced',
                    'change_threshold': change_threshold,
                    'image_bounds': image_bounds or permit_manager.image_bounds,
                    'gps_calculator_stats': gps_stats,
                    'detection_method': 'region_proposal'
                }
            )

            db_manager.save_results(
                session_id=session_id,
                buildings=buildings,
                statistics=statistics,
                processing_time=0
            )

            db_manager.update_session_status(session_id, 'completed')

            # ===== STEP 9: GET USER LOCATION (for front-end view) =====
            public_ip = get_public_ip()
            user_lat, user_lon = 0, 0
            if public_ip:
                user_lat, user_lon = get_location_by_ip(public_ip)

            # Store results in session cache
            session_results[session_id] = {
                'buildings': buildings,
                'statistics': statistics,
                'images': {
                    't1': t1_save_path,
                    't2': t2_save_path,
                    'change': change_path,
                    'detection': detection_path,
                    'illegal': illegal_path
                },
                'gps_stats': gps_stats
            }

            logger.info("=" * 80)
            logger.info(f"✅ Detection completed for session: {session_id}")
            logger.info("=" * 80)

            # ===== RENDER RESULTS (HTML view) =====
            gps = {}

            return render_template(
                'result.html',
                session_id=session_id,
                buildings=buildings,
                gps=gps,
                illegal_buildings=illegal,
                legal_buildings=legal,
                under_construction=under_construction,
                existing_buildings=existing,
                statistics=statistics,
                change_map=f'/results/{os.path.basename(change_path)}',
                detection_overlay=f'/results/{os.path.basename(detection_path)}',
                illegal_overlay=f'/results/{os.path.basename(illegal_path)}' if illegal_path else None,
                t1_image=f'/results/{os.path.basename(t1_save_path)}',
                t2_image=f'/results/{os.path.basename(t2_save_path)}',
                user_lat=user_lat,
                user_lon=user_lon,
                image_bounds=image_bounds or permit_manager.image_bounds,
                change_threshold=change_threshold,
                gps_enabled=True,
                gps_stats=gps_stats
                
            )

        except Exception as e:
            logger.error(f"❌ Detection failed: {e}", exc_info=True)
            flash(f'Detection failed: {str(e)}', 'error')
            return redirect(url_for('upload_page'))

    return redirect(url_for('upload_page'))


@app.route("/api/predict", methods=['POST'])
def api_predict():
    """
    API version of predict that returns JSON instead of HTML
    """
    if request.method == 'POST':
        try:
            # Get uploaded files
            file1 = request.files.get('file1')  # Time 1 (older image)
            file2 = request.files.get('file2')  # Time 2 (newer image)

            if not file1 or not file2:
                # Use HTTP 400 for bad request (missing files)
                return jsonify({
                    'success': False,
                    'error': 'Please upload both temporal images (file1 and file2)'
                }), 400

            # Validate file types
            if not allowed_file(file1.filename) or not allowed_file(file2.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type. Please upload images only.'
                }), 400

            # Save uploaded files
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename1 = f"t1_{timestamp}_{filename1}"
            filename2 = f"t2_{timestamp}_{filename2}"

            file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

            file1.save(file1_path)
            file2.save(file2_path)

            logger.info(f"📤 Uploaded files for API: {file1_path}, {file2_path}")

            # Generate session ID
            session_id = f"session_{uuid.uuid4().hex[:8]}"

            # Get optional GPS bounds (if provided via JSON body or form data)
            image_bounds = None
            try:
                # Try getting bounds from JSON body first (common for API)
                json_data = request.get_json(silent=True)
                if json_data:
                    north = json_data.get('north', type=float)
                    south = json_data.get('south', type=float)
                    east = json_data.get('east', type=float)
                    west = json_data.get('west', type=float)
                else:
                    # Fallback to form data
                    north = request.form.get('north', type=float)
                    south = request.form.get('south', type=float)
                    east = request.form.get('east', type=float)
                    west = request.form.get('west', type=float)

                if all([north, south, east, west]):
                    image_bounds = {
                        'north': north, 'south': south,
                        'east': east, 'west': west
                    }
                    logger.info(f"🗺️  Using API GPS bounds: {image_bounds}")
            except:
                logger.info("📍 No GPS bounds provided via API, using defaults")
            
            # Get change threshold
            change_threshold = float(request.form.get('threshold', 30.0))
            if json_data and 'threshold' in json_data:
                change_threshold = float(json_data['threshold'])
                
            logger.info(f"📊 Using API change threshold: {change_threshold}%")

            # ===== STEP 1: LOAD IMAGES =====
            logger.info("=" * 80)
            logger.info(f"🚀 Starting API Enhanced Detection for session: {session_id}")
            logger.info("=" * 80)
            logger.info("Step 1: Loading images...")

            img1 = load_image_rgb(file1_path)
            img2 = load_image_rgb(file2_path)

            # Resize images
            img1 = resize_keep_aspect(img1, MAX_IMAGE_DIM)
            img2 = resize_keep_aspect(img2, MAX_IMAGE_DIM)

            # Match sizes
            img1, img2 = match_image_sizes(img1, img2, method='crop')
            h, w = img2.shape[:2]
            logger.info(f"✅ Images prepared: {w}x{h}")

            # ===== STEP 2: K-MEANS CHANGE DETECTION =====
            logger.info("Step 2: Running K-means change detection...")
            change_mask = kmeans_change_detection(
                img1, img2,
                k=2,
                downsample=2,
                preprocess=True
            )

            change_stats = get_change_statistics(change_mask)
            logger.info(f"✅ Change detected: {change_stats['change_percentage']:.2f}%")
            logger.info(f"   Changed pixels: {change_stats['changed_pixels']:,}")
            logger.info(f"   Change regions: {change_stats['num_regions']}")

            # ===== STEP 3: INITIALIZE GPS CALCULATOR =====
            logger.info("Step 3: Initializing GPS calculator...")

            # Set image bounds in permit manager
            if image_bounds:
                permit_manager.set_image_bounds(
                    image_bounds['north'],
                    image_bounds['south'],
                    image_bounds['east'],
                    image_bounds['west']
                )

            # Create GPS calculator
            gps_calculator = GPSCoordinateCalculator(
                image_bounds=image_bounds or permit_manager.image_bounds,
                image_size=(w, h)
            )

            # Log GPS statistics
            gps_stats = gps_calculator.get_statistics()
            logger.info(f"✅ GPS Calculator initialized:")
            logger.info(f"   Resolution: {gps_stats['resolution']['meters_per_pixel_x']:.2f}m/px (X), "
                        f"{gps_stats['resolution']['meters_per_pixel_y']:.2f}m/px (Y)")
            logger.info(f"   Coverage: {gps_stats['coverage_area_km2']:.3f} km²")

            # ===== STEP 4: BUILDING DETECTION WITH SEGMENTATION =====
            logger.info("Step 4: Running building segmentation...")
            seg_mask = detector.segment_image(img2)
            building_pixels = np.sum(seg_mask > 0)
            logger.info(f"✅ Building pixels detected: {building_pixels:,}")

            # ===== STEP 5: EXTRACT BUILDINGS WITH REGION PROPOSALS + GPS =====
            logger.info("Step 5: Extracting buildings with ENHANCED DETECTION...")

            buildings = extract_building_metadata(
                seg_mask,
                img2,
                change_mask,
                permit_manager,
                w, h,
                change_threshold,
                gps_calculator=gps_calculator
            )

            # Categorize buildings
            under_construction = [b for b in buildings if b['status'] == 'UNDER_CONSTRUCTION']
            existing = [b for b in buildings if b['status'] == 'EXISTING']
            illegal = [b for b in buildings if b.get('legal_status') == 'ILLEGAL']
            legal = [b for b in buildings if b.get('legal_status') == 'LEGAL']

            logger.info(f"✅ API Buildings categorized:")
            logger.info(f"   Under construction: {len(under_construction)}")
            logger.info(f"   Existing: {len(existing)}")
            logger.info(f"   Illegal: {len(illegal)}")
            logger.info(f"   Legal: {len(legal)}")

            # ===== STEP 6: GENERATE VISUALIZATIONS =====
            logger.info("Step 6: Generating visualizations...")

            # Save original images
            t1_save_path = os.path.join(app.config['RESULTS_FOLDER'], f"t1_{session_id}.png")
            t2_save_path = os.path.join(app.config['RESULTS_FOLDER'], f"t2_{session_id}.png")
            save_image(img1, t1_save_path)
            save_image(img2, t2_save_path)

            # Save change mask
            change_viz = cv2.applyColorMap(change_mask, cv2.COLORMAP_HOT)
            change_path = os.path.join(app.config['RESULTS_FOLDER'], f"change_{session_id}.png")
            save_image(change_viz, change_path)

            # Create detection overlay
            detection_overlay = create_visualization_overlay(img2, buildings, change_mask)
            detection_path = os.path.join(app.config['RESULTS_FOLDER'], f"detection_{session_id}.png")
            save_image(detection_overlay, detection_path)

            # Create illegal buildings overlay
            illegal_path = None
            if illegal:
                illegal_overlay = create_illegal_buildings_overlay(img2, illegal)
                illegal_path = os.path.join(app.config['RESULTS_FOLDER'], f"illegal_{session_id}.png")
                save_image(illegal_overlay, illegal_path)

            logger.info(f"✅ API Visualizations saved")

            # ===== STEP 7: CALCULATE STATISTICS =====
            statistics = {
                'total_pixels': int(h * w),
                'change_pixels': int(change_stats['changed_pixels']),
                'change_percentage': float(change_stats['change_percentage']),
                'num_change_regions': int(change_stats['num_regions']),
                'total_buildings': len(buildings),
                'buildings_under_construction': len(under_construction),
                'existing_buildings': len(existing),
                'legal_constructions': len(legal),
                'illegal_constructions': len(illegal),
                'total_building_area_m2': int(sum(b.get('area_m2', 0) for b in buildings)),
                'illegal_building_area_m2': int(sum(b.get('area_m2', 0) for b in illegal)),
                'compliance_rate': round((len(legal) / len(under_construction) * 100), 2) if under_construction else 100.0,
                'change_threshold': change_threshold,
                'gps_resolution_m_per_px': round(gps_stats['resolution']['meters_per_pixel_x'], 3),
                'coverage_area_km2': gps_stats['coverage_area_km2'],
                'detection_method': 'region_proposal + multi_scale'
            }

            # ===== STEP 8: SAVE TO DATABASE =====
            logger.info("Step 7: Saving to database...")
            db_manager.create_session(
                session_id=session_id,
                t1_path=file1_path,
                t2_path=file2_path,
                metadata={
                    'mode': 'two-temporal-kmeans-gps-enhanced',
                    'change_threshold': change_threshold,
                    'image_bounds': image_bounds or permit_manager.image_bounds,
                    'gps_calculator_stats': gps_stats,
                    'detection_method': 'region_proposal'
                }
            )

            db_manager.save_results(
                session_id=session_id,
                buildings=buildings,
                statistics=statistics,
                processing_time=0
            )

            db_manager.update_session_status(session_id, 'completed')

            # Store results in session cache
            session_results[session_id] = {
                'buildings': buildings,
                'statistics': statistics,
                'images': {
                    't1': t1_save_path,
                    't2': t2_save_path,
                    'change': change_path,
                    'detection': detection_path,
                    'illegal': illegal_path
                },
                'gps_stats': gps_stats
            }

            logger.info("=" * 80)
            logger.info(f"✅ API Detection completed for session: {session_id}")
            logger.info("=" * 80)

            # ===== RETURN JSON RESPONSE =====
            return jsonify({
                'success': True,
                'session_id': session_id,
                'buildings': buildings,
                'statistics': statistics,
                'images': {
                    # Note: These are only the filenames, clients must call /results/<filename>
                    't1': os.path.basename(t1_save_path),
                    't2': os.path.basename(t2_save_path),
                    'change': os.path.basename(change_path),
                    'detection': os.path.basename(detection_path),
                    'illegal': os.path.basename(illegal_path) if illegal_path else None
                },
                'gps_stats': gps_stats
            }), 200

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500


@app.route('/results/<filename>')
def serve_results(filename):
    """Serve result images"""
    try:
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        return send_file(filepath, mimetype='image/png')
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return "File not found", 404


@app.route('/api/generate-report/<session_id>', methods=['GET'])
def generate_report(session_id):
    """Generate PDF report for session"""
    try:
        logger.info(f"📄 Generating report for session: {session_id}")

        # Get results from cache or database
        if session_id in session_results:
            results = session_results[session_id]
        else:
            db_results = db_manager.get_results(session_id)
            if not db_results or not isinstance(db_results, dict):
                return jsonify({'error': 'Session data not found'}), 404
            results = db_results

        if not results:
            return jsonify({'error': 'Session not found'}), 404

        # Prepare images dictionary with absolute paths
        images_dict = {}
        images = results.get('images', {})

        if images.get('t1'):
            images_dict['Time 1 (Older) Image'] = images['t1']
        if images.get('t2'):
            images_dict['Time 2 (Newer) Image'] = images['t2']
        if images.get('change'):
            images_dict['K-means Change Detection Map'] = images['change']
        if images.get('detection'):
            images_dict['Building Detection Overlay'] = images['detection']
        if images.get('illegal'):
            images_dict['Illegal Buildings Only'] = images['illegal']

        # Generate report
        report_path = report_generator.generate_report(
            session_id=session_id,
            results=results,
            images_dict=images_dict
        )

        logger.info(f"✅ Report generated: {report_path}")

        return send_file(
            report_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"BuildingDetectionReport_{session_id[:8]}.pdf"
        )

    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/export-geojson/<session_id>', methods=['GET'])
def export_geojson(session_id):
    """Export buildings as GeoJSON with accurate GPS polygons"""
    try:
        logger.info(f"🗺️ Exporting GeoJSON for session: {session_id}")

        # Get results
        if session_id in session_results:
            results = session_results[session_id]
        else:
            results = db_manager.get_results(session_id)

        if not results:
            return jsonify({'error': 'Session not found'}), 404

        buildings = results.get('buildings', [])

        # Create GeoJSON FeatureCollection
        features = []

        for building in buildings:
            # Prefer polygon representation
            if 'geojson_polygon' in building:
                features.append(building['geojson_polygon'])
            elif 'geojson_point' in building:
                features.append(building['geojson_point'])
            else:
                # Fallback: create point from lat/lon
                lat = building.get('gps_latitude')
                lon = building.get('gps_longitude')
                if lat and lon:
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat]
                        },
                        "properties": {
                            "building_id": building.get('id'),
                            "status": building.get('legal_status', 'UNKNOWN'),
                            "area_m2": building.get('area_m2', 0),
                            "perimeter_m": building.get('perimeter_m', 0),
                            "change_percentage": building.get('change_percentage', 0),
                            "confidence": building.get('confidence', 0),
                            "construction_status": building.get('status', 'UNKNOWN'),
                            "detection_method": building.get('detection_method', 'unknown')
                        }
                    })

        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "EPSG:4326"}
            },
            "features": features,
            "metadata": {
                "session_id": session_id,
                "total_buildings": len(buildings),
                "illegal_buildings": len([b for b in buildings if b.get('legal_status') == 'ILLEGAL']),
                "legal_buildings": len([b for b in buildings if b.get('legal_status') == 'LEGAL']),
                "generated_at": datetime.now().isoformat(),
                "coordinate_system": "WGS84 (EPSG:4326)",
                "detection_method": "region_proposal + multi_scale",
                "gps_calculator": results.get('gps_stats', {})
            }
        }

        # Save to file
        output_path = os.path.join(
            app.config['RESULTS_FOLDER'],
            f'buildings_{session_id}.geojson'
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2)

        logger.info(f"✅ GeoJSON exported: {len(features)} features")

        return send_file(
            output_path,
            mimetype='application/geo+json',
            as_attachment=True,
            download_name=f'buildings_{session_id[:8]}.geojson'
        )

    except Exception as e:
        logger.error(f"❌ GeoJSON export failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': 'connected',
        'detector': 'ready',
        'features': {
            'change_detection': 'K-means Clustering',
            'building_detection': 'Region Proposals + Multi-Scale',
            'gps_coordinates': 'Enhanced (Bilinear + Mercator)',
            'accurate_area_calculation': True,
            'accurate_bounding_boxes': True,
            'nms_filtering': True,
            'geojson_export': True,
            'pdf_reports': True
        },
        'version': '3.0-enhanced'
    }), 200


@app.route('/api/detection-stats')
def get_detection_stats():
    """Get detection method statistics"""
    try:
        total_sessions = len(session_results)

        total_buildings = 0
        total_with_region_proposals = 0
        avg_confidence = []

        for session_id, results in session_results.items():
            buildings = results.get('buildings', [])
            total_buildings += len(buildings)

            for building in buildings:
                if building.get('detection_method') == 'region_proposal':
                    total_with_region_proposals += 1
                avg_confidence.append(building.get('confidence', 0))

        return jsonify({
            'total_sessions': total_sessions,
            'total_buildings': total_buildings,
            'region_proposal_detections': total_with_region_proposals,
            'average_confidence': round(np.mean(avg_confidence), 3) if avg_confidence else 0,
            'detection_method': 'Multi-Scale Region Proposals with NMS'
        }), 200

    except Exception as e:
        logger.error(f"Error getting detection stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# ========================================================================
# ADD THESE ROUTES TO YOUR app.py FILE
# Insert after the existing routes (before if __name__ == "__main__")
# ========================================================================

from field_visit_manager import FieldVisitManager

# Initialize field visit manager
field_visit_manager = FieldVisitManager(DATABASE_PATH)


# ========================================================================
# FIELD VISIT ROUTES
# ========================================================================

@app.route('/api/field-visits/create', methods=['POST'])
def create_field_visit():
    """
    Create a new field visit request

    POST Body:
    {
        "building_id": "BLDG_12345678",
        "session_id": "session_12345678",
        "inspector_name": "John Doe",
        "priority": "high",
        "scheduled_date": "2025-01-20",
        "gps_latitude": 12.3000,
        "gps_longitude": 76.6500,
        "address": "123 Main St",
        "notes": "Suspected illegal construction"
    }
    """
    try:
        data = request.get_json()

        visit_id = field_visit_manager.create_field_visit(
            building_id=data['building_id'],
            session_id=data.get('session_id'),
            inspector_name=data.get('inspector_name'),
            priority=data.get('priority', 'medium'),
            scheduled_date=data.get('scheduled_date'),
            gps_latitude=data.get('gps_latitude'),
            gps_longitude=data.get('gps_longitude'),
            address=data.get('address'),
            notes=data.get('notes')
        )

        return jsonify({
            'success': True,
            'visit_id': visit_id,
            'message': 'Field visit created successfully'
        }), 201

    except Exception as e:
        logger.error(f"Error creating field visit: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/field-visits/<visit_id>', methods=['GET'])
def get_field_visit(visit_id):
    """Get field visit details"""
    try:
        visit = field_visit_manager.get_field_visit(visit_id)

        if not visit:
            return jsonify({'error': 'Visit not found'}), 404

        return jsonify({
            'success': True,
            'visit': visit
        }), 200

    except Exception as e:
        logger.error(f"Error getting field visit: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/field-visits/pending', methods=['GET'])
def get_pending_visits():
    """Get all pending field visits"""
    try:
        priority = request.args.get('priority')
        visits = field_visit_manager.get_pending_visits(priority=priority)

        return jsonify({
            'success': True,
            'count': len(visits),
            'visits': visits
        }), 200

    except Exception as e:
        logger.error(f"Error getting pending visits: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/field-visits/<visit_id>/status', methods=['PUT'])
def update_visit_status(visit_id):
    """
    Update field visit status

    PUT Body:
    {
        "status": "completed",
        "inspector_id": "INS123"
    }
    """
    try:
        data = request.get_json()

        field_visit_manager.update_visit_status(
            visit_id=visit_id,
            status=data['status'],
            inspector_id=data.get('inspector_id')
        )

        return jsonify({
            'success': True,
            'message': f'Visit status updated to {data["status"]}'
        }), 200

    except Exception as e:
        logger.error(f"Error updating visit status: {e}")
        return jsonify({'error': str(e)}), 500


# ========================================================================
# VIOLATION ROUTES
# ========================================================================

@app.route('/api/violations/report', methods=['POST'])
def report_violation():
    """
    Report a violation found during field visit

    POST Body:
    {
        "visit_id": "VISIT_12345678",
        "building_id": "BLDG_12345678",
        "violation_type": "no_permit",
        "severity": "high",
        "description": "Building has no valid permit",
        "evidence_photos": ["/path/to/photo1.jpg", "/path/to/photo2.jpg"],
        "deadline": "2025-02-15"
    }
    """
    try:
        data = request.get_json()

        violation_id = field_visit_manager.report_violation(
            visit_id=data['visit_id'],
            building_id=data['building_id'],
            violation_type=data['violation_type'],
            severity=data.get('severity', 'medium'),
            description=data.get('description'),
            evidence_photos=data.get('evidence_photos'),
            deadline=data.get('deadline')
        )

        return jsonify({
            'success': True,
            'violation_id': violation_id,
            'message': 'Violation reported successfully'
        }), 201

    except Exception as e:
        logger.error(f"Error reporting violation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/violations/open', methods=['GET'])
def get_open_violations():
    """Get all open violations"""
    try:
        building_id = request.args.get('building_id')
        violations = field_visit_manager.get_open_violations(building_id=building_id)

        return jsonify({
            'success': True,
            'count': len(violations),
            'violations': violations
        }), 200

    except Exception as e:
        logger.error(f"Error getting violations: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/violations/<violation_id>/resolve', methods=['PUT'])
def resolve_violation(violation_id):
    """
    Mark violation as resolved

    PUT Body:
    {
        "corrective_action": "Owner obtained valid permit"
    }
    """
    try:
        data = request.get_json()

        field_visit_manager.resolve_violation(
            violation_id=violation_id,
            corrective_action=data.get('corrective_action')
        )

        return jsonify({
            'success': True,
            'message': 'Violation resolved successfully'
        }), 200

    except Exception as e:
        logger.error(f"Error resolving violation: {e}")
        return jsonify({'error': str(e)}), 500


# ========================================================================
# INSPECTION ROUTES
# ========================================================================

@app.route('/api/inspections/<visit_id>/items', methods=['POST'])
def add_inspection_item(visit_id):
    """
    Add an inspection checklist item

    POST Body:
    {
        "item_name": "Building Permit",
        "category": "Documentation",
        "compliant": false,
        "notes": "No permit found"
    }
    """
    try:
        data = request.get_json()

        field_visit_manager.add_inspection_item(
            visit_id=visit_id,
            item_name=data['item_name'],
            category=data.get('category'),
            compliant=data.get('compliant'),
            notes=data.get('notes')
        )

        return jsonify({
            'success': True,
            'message': 'Inspection item added successfully'
        }), 201

    except Exception as e:
        logger.error(f"Error adding inspection item: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/inspections/<visit_id>/checklist', methods=['GET'])
def get_inspection_checklist(visit_id):
    """Get inspection checklist for a visit"""
    try:
        checklist = field_visit_manager.get_inspection_checklist(visit_id)

        return jsonify({
            'success': True,
            'count': len(checklist),
            'items': checklist
        }), 200

    except Exception as e:
        logger.error(f"Error getting checklist: {e}")
        return jsonify({'error': str(e)}), 500


# ========================================================================
# FIELD PHOTOS ROUTES
# ========================================================================

@app.route('/api/field-photos/upload', methods=['POST'])
def upload_field_photo():
    """
    Upload a field photo

    Form Data:
        visit_id: Visit ID
        photo: Image file
        photo_type: Type of photo
        description: Description
        gps_latitude: GPS latitude
        gps_longitude: GPS longitude
    """
    try:
        if 'photo' not in request.files:
            return jsonify({'error': 'No photo file provided'}), 400

        file = request.files['photo']
        visit_id = request.form.get('visit_id')

        if not visit_id:
            return jsonify({'error': 'visit_id is required'}), 400

        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"field_{visit_id}_{timestamp}_{filename}"

        photo_dir = os.path.join(app.config['RESULTS_FOLDER'], 'field_photos')
        os.makedirs(photo_dir, exist_ok=True)

        file_path = os.path.join(photo_dir, filename)
        file.save(file_path)

        # Add to database
        photo_id = field_visit_manager.add_field_photo(
            visit_id=visit_id,
            file_path=file_path,
            photo_type=request.form.get('photo_type'),
            description=request.form.get('description'),
            gps_latitude=request.form.get('gps_latitude', type=float),
            gps_longitude=request.form.get('gps_longitude', type=float)
        )

        return jsonify({
            'success': True,
            'photo_id': photo_id,
            'file_path': file_path,
            'message': 'Photo uploaded successfully'
        }), 201

    except Exception as e:
        logger.error(f"Error uploading photo: {e}")
        return jsonify({'error': str(e)}), 500


# ========================================================================
# FOLLOW-UP ACTIONS ROUTES
# ========================================================================

@app.route('/api/follow-up-actions/create', methods=['POST'])
def create_follow_up_action():
    """
    Create a follow-up action

    POST Body:
    {
        "visit_id": "VISIT_12345678",
        "violation_id": "VIOL_12345678",
        "action_type": "legal_notice",
        "description": "Send notice to owner",
        "assigned_to": "Legal Team",
        "due_date": "2025-02-01"
    }
    """
    try:
        data = request.get_json()

        action_id = field_visit_manager.create_follow_up_action(
            visit_id=data['visit_id'],
            action_type=data['action_type'],
            description=data['description'],
            assigned_to=data.get('assigned_to'),
            due_date=data.get('due_date'),
            violation_id=data.get('violation_id')
        )

        return jsonify({
            'success': True,
            'action_id': action_id,
            'message': 'Follow-up action created successfully'
        }), 201

    except Exception as e:
        logger.error(f"Error creating follow-up action: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/follow-up-actions/<action_id>/complete', methods=['PUT'])
def complete_follow_up_action(action_id):
    """
    Mark follow-up action as completed

    PUT Body:
    {
        "notes": "Notice delivered successfully"
    }
    """
    try:
        data = request.get_json()

        field_visit_manager.complete_follow_up_action(
            action_id=action_id,
            notes=data.get('notes')
        )

        return jsonify({
            'success': True,
            'message': 'Follow-up action completed successfully'
        }), 200

    except Exception as e:
        logger.error(f"Error completing action: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cadastral/import', methods=['POST'])
def import_cadastral_data():
    """
    Import cadastral shapefile

    POST Body:
    {
        "file_path": "/path/to/permits.shp",
        "layer_type": "permits"
    }
    """
    try:
        data = request.get_json()

        result = cadastral_integration.import_cadastral_shapefile(
            shapefile_path=data['file_path'],
            layer_type=data.get('layer_type', 'permits')
        )

        return jsonify({
            'success': True,
            'imported': result['imported'],
            'failed': result['failed'],
            'total': result['total']
        }), 200

    except Exception as e:
        logger.error(f"Cadastral import failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cadastral/match-buildings/<session_id>', methods=['POST'])
def match_buildings_to_cadastral(session_id):
    """Match all buildings in a session to cadastral permits"""
    try:
        # Get session results
        if session_id in session_results:
            results = session_results[session_id]
        else:
            results = db_manager.get_results(session_id)

        if not results:
            return jsonify({'error': 'Session not found'}), 404

        buildings = results.get('buildings', [])

        # Batch match
        match_results = cadastral_integration.batch_match_buildings(buildings)

        return jsonify({
            'success': True,
            'results': match_results
        }), 200

    except Exception as e:
        logger.error(f"Cadastral matching failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cadastral/statistics', methods=['GET'])
def get_cadastral_statistics():
    """Get cadastral database statistics"""
    try:
        stats = cadastral_integration.get_statistics()

        return jsonify({
            'success': True,
            'statistics': stats
        }), 200

    except Exception as e:
        logger.error(f"Error getting cadastral stats: {e}")
        return jsonify({'error': str(e)}), 500


# ========================================================================
# STATISTICS & REPORTS ROUTES
# ========================================================================

@app.route('/api/field-visits/statistics', methods=['GET'])
def get_field_visit_statistics():
    """Get field visit statistics"""
    try:
        stats = field_visit_manager.get_visit_statistics()

        return jsonify({
            'success': True,
            'statistics': stats
        }), 200

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/field-visits/<visit_id>/report', methods=['GET'])
def generate_field_visit_report(visit_id):
    """Generate comprehensive field visit report"""
    try:
        report = field_visit_manager.generate_field_report(visit_id)

        if not report:
            return jsonify({'error': 'Visit not found'}), 404

        return jsonify({
            'success': True,
            'report': report
        }), 200

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({'error': str(e)}), 500


# ========================================================================
# BULK OPERATIONS
# ========================================================================

# Add this to your app.py BEFORE the if __name__ == "__main__": section

from werkzeug.security import generate_password_hash, check_password_hash

@app.route('/api/register', methods=['POST', 'OPTIONS'])
def api_register():
    """
    User registration endpoint
    
    POST Body:
    {
        "username": "john_doe",
        "email": "john@example.com",
        "password": "secure_password_123",
        "full_name": "John Doe"
    }
    """
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['username', 'email', 'password']
        missing_fields = [f for f in required_fields if f not in data or not data[f]]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        username = data.get('username').strip()
        email = data.get('email').strip().lower()
        password = data.get('password')
        full_name = data.get('full_name', username)
        
        # Validate inputs
        if len(username) < 3:
            return jsonify({
                'success': False,
                'error': 'Username must be at least 3 characters long'
            }), 400
        
        if len(password) < 6:
            return jsonify({
                'success': False,
                'error': 'Password must be at least 6 characters long'
            }), 400
        
        if '@' not in email:
            return jsonify({
                'success': False,
                'error': 'Invalid email address'
            }), 400
        
        # Check if user already exists in database
        # You'll need to implement this based on your database schema
        # For now, this is a placeholder
        
        # Hash password
        hashed_password = generate_password_hash(password)
        
        # Save user to database (implement based on your schema)
        # Example: db_manager.create_user(username, email, hashed_password, full_name)
        
        logger.info(f"✅ User registered: {username} ({email})")
        
        return jsonify({
            'success': True,
            'message': f'User {username} registered successfully',
            'user': {
                'username': username,
                'email': email,
                'full_name': full_name
            }
        }), 201
    
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/login', methods=['POST', 'OPTIONS'])
def api_login():
    """
    User login endpoint
    
    POST Body:
    {
        "username": "john_doe",
        "password": "secure_password_123"
    }
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({
                'success': False,
                'error': 'Username and password required'
            }), 400
        
        # TODO: Verify user credentials from database
        # user = db_manager.get_user(username)
        # if not user or not check_password_hash(user['password'], password):
        #     return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
        
        logger.info(f"✅ User logged in: {username}")
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'username': username,
                'token': 'your_jwt_token_here'  # Implement JWT if needed
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/field-visits/create-from-session/<session_id>', methods=['POST'])
def create_visits_from_session(session_id):
    """
    Create field visits for all illegal buildings in a session

    POST Body:
    {
        "inspector_name": "John Doe",
        "priority": "high",
        "scheduled_date": "2025-01-20"
    }
    """
    try:
        data = request.get_json()

        # Get session results
        if session_id in session_results:
            results = session_results[session_id]
        else:
            results = db_manager.get_results(session_id)

        if not results:
            return jsonify({'error': 'Session not found'}), 404

        buildings = results.get('buildings', [])
        illegal_buildings = [b for b in buildings if b.get('legal_status') == 'ILLEGAL']

        created_visits = []

        for building in illegal_buildings:
            visit_id = field_visit_manager.create_field_visit(
                building_id=building['id'],
                session_id=session_id,
                inspector_name=data.get('inspector_name'),
                priority=data.get('priority', 'high'),
                scheduled_date=data.get('scheduled_date'),
                gps_latitude=building.get('gps_latitude'),
                gps_longitude=building.get('gps_longitude'),
                address=data.get('address'),
                notes=f"Auto-created for illegal building. {building.get('legal_reason', '')}"
            )
            created_visits.append(visit_id)

        return jsonify({
            'success': True,
            'created_count': len(created_visits),
            'visit_ids': created_visits,
            'message': f'Created {len(created_visits)} field visits for illegal buildings'
        }), 201

    except Exception as e:
        logger.error(f"Error creating bulk visits: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def api_get_session(session_id):
    """
    API endpoint to get session data as JSON
    Used by frontend to fetch results
    """
    try:
        # Get results from database
        results = db_manager.get_results(session_id)

        if not results:
            return jsonify({
                'success': False,
                'error': 'Session not found'
            }), 404

        return jsonify({
            'success': True,
            'session': results.get('session', {}),
            'buildings': results.get('buildings', []),
            'statistics': results.get('statistics', {}),
            'processing_time': results.get('processing_time'),
            'detection_metadata': results.get('detection_metadata', {})
        }), 200

    except Exception as e:
        logger.error(f"Error fetching session: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# APPLICATION STARTUP
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🏗️  ENHANCED ILLEGAL BUILDING DETECTION SYSTEM v3.0")
    logger.info("=" * 80)
    logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
    logger.info(f"📁 Results directory: {RESULTS_DIR}")
    logger.info(f"💾 Database: {DATABASE_PATH}")
    logger.info("")
    logger.info("🔬 DETECTION PIPELINE:")
    logger.info("  1️⃣  K-means Clustering (Change Detection)")
    logger.info("  2️⃣  Region Proposals (Building Detection)")
    logger.info("  3️⃣  Multi-Scale Processing")
    logger.info("  4️⃣  Non-Maximum Suppression (NMS)")
    logger.info("  5️⃣  Confidence Scoring")
    logger.info("  6️⃣  Enhanced GPS Coordinates")
    logger.info("")
    logger.info("📐 FEATURES:")
    logger.info("  ✅ Accurate bounding boxes")
    logger.info("  ✅ Region proposal detection")
    logger.info("  ✅ Multi-scale analysis")
    logger.info("  ✅ GPS coordinates (±2-10m accuracy)")
    logger.info("  ✅ Area calculation (m²)")
    logger.info("  ✅ Perimeter calculation (m)")
    logger.info("  ✅ Building polygon export")
    logger.info("  ✅ GeoJSON with polygons")
    logger.info("  ✅ PDF reports with visualizations")
    logger.info("  ✅ Confidence scores per building")
    logger.info("")
    logger.info("📊 PARAMETERS:")
    logger.info("  • Change Threshold: 30% (Default)")
    logger.info("  • Min Building Area: 50 pixels")
    logger.info("  • NMS Threshold: 0.3 IoU")
    logger.info("  • Confidence Threshold: 0.5")
    logger.info("  • Detection Scales: [0.9, 1.0, 1.1]")
    logger.info("")
    logger.info("🌐 Server: http://localhost:5000")
    logger.info("=" * 80)
    logger.info("")
    logger.info("✅ System Ready!")
    logger.info("  Upload two temporal images to detect illegal buildings")
    logger.info("  Enhanced detection with region proposals active")
    logger.info("")
    logger.info("📡 Available Endpoints:")
    logger.info("  POST   /predict                   - Main detection (HTML)")
    logger.info("  POST   /api/predict               - Main detection (JSON)")
    logger.info("  GET    /api/generate-report/<id>  - Generate PDF")
    logger.info("  GET    /api/export-geojson/<id>   - Export GeoJSON")
    logger.info("  GET    /api/detection-stats       - Detection statistics")
    logger.info("  GET    /api/health                - Health check")
    logger.info("=" * 80)

    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)