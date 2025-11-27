# app.py - INTEGRATED WITH main_two_temporal.py PIPELINE

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

# Keras imports
from tensorflow.keras.models import load_model

# Flask imports
from flask import Flask, redirect, url_for, request, render_template, session, flash, jsonify, send_file
from werkzeug.utils import secure_filename

# ===== IMPORT THE COMPLETE PIPELINE =====
from main_two_temporal import run_two_temporal_pipeline
from config import DATABASE_PATH, UPLOAD_DIR, RESULTS_DIR, SEG_MODEL_DIR, MAX_IMAGE_DIM
from database_manager import DatabaseManager
from permit_manager import PermitManager
from report_generator import PDFReportGenerator

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "illegal_building_detection_secret_key_2024"
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['RESULTS_FOLDER'] = RESULTS_DIR
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff', 'bmp'}

# Initialize components
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

db_manager = DatabaseManager(DATABASE_PATH)
permit_manager = PermitManager(DATABASE_PATH)
report_generator = PDFReportGenerator(output_dir=os.path.join(RESULTS_DIR, 'reports'))

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


def get_web_url(absolute_path):
    """
    Converts an absolute file path to a web-accessible URL
    """
    if absolute_path and absolute_path.startswith(app.config['RESULTS_FOLDER']):
        relative_path = os.path.relpath(absolute_path, app.config['RESULTS_FOLDER'])
        return f'/results/{relative_path.replace(os.path.sep, "/")}'
    return ''


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
    Main prediction endpoint using K-MEANS CLUSTERING
    Fully integrated with main_two_temporal.py pipeline
    """
    if request.method == 'POST':
        try:
            # ===== STEP 1: VALIDATE AND SAVE UPLOADED FILES =====
            file1 = request.files.get('file1')  # Time 1 (older image)
            file2 = request.files.get('file2')  # Time 2 (newer image)
            
            if not file1 or not file2:
                flash('Please upload both temporal images (Time 1 and Time 2)', 'error')
                return redirect(url_for('upload_page'))
            
            # Validate file types
            if not allowed_file(file1.filename) or not allowed_file(file2.filename):
                flash('Invalid file type. Allowed: PNG, JPG, JPEG, GIF, TIF, TIFF, BMP', 'error')
                return redirect(url_for('upload_page'))
            
            # Save uploaded files with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            
            filename1 = f"t1_{timestamp}_{filename1}"
            filename2 = f"t2_{timestamp}_{filename2}"
            
            file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            
            file1.save(file1_path)
            file2.save(file2_path)
            
            logger.info(f"✅ Uploaded files saved:")
            logger.info(f"   Time 1: {file1_path}")
            logger.info(f"   Time 2: {file2_path}")
            
            # ===== STEP 2: GET OPTIONAL PARAMETERS =====
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
                        'north': north, 
                        'south': south,
                        'east': east, 
                        'west': west
                    }
                    logger.info(f"🌍 Using custom GPS bounds: {image_bounds}")
                else:
                    logger.info("📍 No GPS bounds provided, using defaults")
            except Exception as e:
                logger.warning(f"Error parsing GPS bounds: {e}")
                logger.info("📍 Using default GPS bounds")
            
            # Get change threshold (default 30% as per PDF)
            change_threshold = 30.0
            try:
                change_threshold = float(request.form.get('threshold', 30.0))
                logger.info(f"📊 Using change threshold: {change_threshold}%")
            except:
                logger.warning("Invalid threshold value, using default 30%")
            
            # ===== STEP 3: RUN THE COMPLETE TWO-TEMPORAL PIPELINE =====
            logger.info("=" * 80)
            logger.info(f"🚀 Starting K-means based detection for session: {session_id}")
            logger.info("=" * 80)
            
            # Call the main_two_temporal.py pipeline
            results = run_two_temporal_pipeline(
                image_t1_path=file1_path,
                image_t2_path=file2_path,
                uid=session_id,
                results_dir=app.config['RESULTS_FOLDER'],
                image_bounds=image_bounds,
                change_threshold=change_threshold
            )
            
            logger.info("=" * 80)
            logger.info(f"✅ Detection pipeline completed successfully!")
            logger.info("=" * 80)
            
            # ===== STEP 4: EXTRACT RESULTS =====
            buildings = results.get('buildings', [])
            statistics = results.get('statistics', {})
            
            # Categorize buildings
            under_construction = results.get('under_construction', [])
            existing_buildings = results.get('existing_buildings', [])
            illegal_buildings = results.get('illegal_buildings', [])
            legal_buildings = results.get('legal_buildings', [])
            
            # Get image paths (convert to web URLs)
            images = results.get('images', {})
            t1_image = get_web_url(images.get('t1_original', ''))
            t2_image = get_web_url(images.get('t2_original', ''))
            change_map = get_web_url(images.get('change_mask', ''))
            detection_overlay = get_web_url(images.get('detection_overlay', ''))
            illegal_overlay = get_web_url(images.get('illegal_only', ''))
            
            logger.info(f"📊 Detection Summary:")
            logger.info(f"   Total Buildings: {len(buildings)}")
            logger.info(f"   Under Construction: {len(under_construction)}")
            logger.info(f"   Existing: {len(existing_buildings)}")
            logger.info(f"   Illegal: {len(illegal_buildings)}")
            logger.info(f"   Legal: {len(legal_buildings)}")
            logger.info(f"   Change %: {statistics.get('change_percentage', 0):.2f}%")
            
            # ===== STEP 5: SAVE TO DATABASE =====
            try:
                db_manager.create_session(
                    session_id=session_id,
                    t1_path=file1_path,
                    t2_path=file2_path,
                    metadata={
                        'mode': 'two-temporal-kmeans',
                        'change_threshold': change_threshold,
                        'image_bounds': image_bounds or results.get('image_bounds'),
                        'timestamp': timestamp
                    }
                )
                
                db_manager.save_results(
                    session_id=session_id,
                    buildings=buildings,
                    statistics=statistics,
                    processing_time=float(results.get('processing_time', '0s').replace('s', ''))
                )
                
                db_manager.update_session_status(session_id, 'completed')
                logger.info(f"💾 Results saved to database for session: {session_id}")
            except Exception as e:
                logger.error(f"Database save failed: {e}")
            
            # ===== STEP 6: GET USER LOCATION (FOR MAP) =====
            public_ip = get_public_ip()
            user_lat, user_lon = 0, 0
            if public_ip:
                user_lat, user_lon = get_location_by_ip(public_ip)
                logger.info(f"📍 User location: ({user_lat}, {user_lon})")
            
            # Store results in session cache for report generation
            session_results[session_id] = results
            
            # ===== STEP 7: RENDER RESULTS PAGE =====
            return render_template(
                'result.html',
                # Session info
                session_id=session_id,
                
                # Buildings data
                buildings=buildings,
                illegal_buildings=illegal_buildings,
                legal_buildings=legal_buildings,
                under_construction=under_construction,
                existing_buildings=existing_buildings,
                
                # Statistics
                statistics=statistics,
                
                # Images (web URLs)
                t1_image=t1_image,
                t2_image=t2_image,
                change_map=change_map,
                detection_overlay=detection_overlay,
                illegal_overlay=illegal_overlay if illegal_overlay else None,
                
                # GPS and parameters
                user_lat=user_lat,
                user_lon=user_lon,
                image_bounds=results.get('image_bounds'),
                change_threshold=change_threshold,
                
                # Processing info
                processing_time=results.get('processing_time', '0s'),
                gps_enabled=results.get('gps_enabled', False),
                
                # Method info
                detection_method='K-means Clustering',
                pdf_compliant=results.get('pdf_compliant', True)
            )
            
        except Exception as e:
            logger.error(f"❌ Detection failed: {e}", exc_info=True)
            flash(f'Detection failed: {str(e)}', 'error')
            return redirect(url_for('upload_page'))
    
    return redirect(url_for('upload_page'))


@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve result images and files"""
    try:
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        
        # Determine MIME type
        ext = filename.rsplit('.', 1)[-1].lower()
        if ext in ['jpg', 'jpeg']:
            mimetype = 'image/jpeg'
        elif ext == 'gif':
            mimetype = 'image/gif'
        else:
            mimetype = 'image/png'
        
        return send_file(filepath, mimetype=mimetype)
        
    except FileNotFoundError:
        logger.error(f"❌ File not found: {filename}")
        return "File not found", 404
    except Exception as e:
        logger.error(f"❌ Error serving file {filename}: {e}")
        return "Error serving file", 500


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
        
        if images.get('t1_original'):
            images_dict['Time 1 (Older) Image'] = images['t1_original']
        if images.get('t2_original'):
            images_dict['Time 2 (Newer) Image'] = images['t2_original']
        if images.get('change_mask'):
            images_dict['K-means Change Detection Map'] = images['change_mask']
        if images.get('detection_overlay'):
            images_dict['Building Detection Overlay'] = images['detection_overlay']
        if images.get('illegal_only'):
            images_dict['Illegal Buildings Only'] = images['illegal_only']
        
        # Log image paths for debugging
        logger.info("📷 Images for report:")
        for desc, path in images_dict.items():
            exists = os.path.exists(path) if path else False
            logger.info(f"   {desc}: {path} (Exists: {exists})")
        
        # Generate report
        report_path = report_generator.generate_report(
            session_id=session_id,
            results=results,
            images_dict=images_dict
        )
        
        # Verify report was created
        if not os.path.exists(report_path):
            raise IOError(f"Report was not created: {report_path}")
        
        file_size = os.path.getsize(report_path)
        size_mb = round(file_size / (1024 * 1024), 2)
        
        logger.info(f"✅ Report generated: {report_path} ({size_mb} MB)")
        
        return send_file(
            report_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"BuildingDetectionReport_{session_id[:8]}.pdf"
        )
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}", exc_info=True)
        return jsonify({
            'error': f'Report generation failed: {str(e)}'
        }), 500


@app.route('/api/session/<session_id>')
def get_session_details(session_id):
    """Get detailed information about a session"""
    try:
        results = db_manager.get_results(session_id)
        
        if not results:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify({
            'success': True,
            'session': results.get('session', {}),
            'buildings': results.get('buildings', []),
            'statistics': results.get('statistics', {})
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting session details: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions/list')
def list_sessions():
    """List all processing sessions"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, created_at, status, metadata 
                FROM sessions 
                ORDER BY created_at DESC 
                LIMIT 50
            ''')
            sessions = [dict(zip([column[0] for column in cursor.description], row)) 
                       for row in cursor.fetchall()]
        
        return jsonify({
            'success': True,
            'sessions': sessions,
            'count': len(sessions)
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error listing sessions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': 'connected',
        'detector': 'ready',
        'method': 'K-means Clustering',
        'modes': ['two-temporal'],
        'version': '2.0',
        'pdf_compliant': True
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# APPLICATION STARTUP
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🏗️  GPS-Enabled Illegal Building Detection System")
    logger.info("=" * 80)
    logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
    logger.info(f"📁 Results directory: {RESULTS_DIR}")
    logger.info(f"💾 Database: {DATABASE_PATH}")
    logger.info("🔬 Detection Method: K-means Clustering (PDF Compliant)")
    logger.info("📊 Change Threshold: 30% (Default)")
    logger.info("🌍 GPS Integration: Enabled")
    logger.info("📄 Report Generation: Enabled")
    logger.info("🌐 Server: http://localhost:5000")
    logger.info("=" * 80)
    logger.info("")
    logger.info("✅ System Ready!")
    logger.info("   Upload two temporal images to detect illegal buildings")
    logger.info("   Uses K-means clustering for change detection")
    logger.info("   Applies 30% threshold as per project specifications")
    logger.info("")
    logger.info("=" * 80)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)