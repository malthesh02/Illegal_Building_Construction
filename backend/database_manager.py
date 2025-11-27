"""
Enhanced Database Manager with Complete Integration
Properly saves sessions, results, permits, and metadata

Author: Enhanced Version
Date: 2025
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class EnhancedDatabaseManager:
    """
    Enhanced Database Manager with complete integration for:
    - Sessions (detection runs)
    - Results (statistics and summaries)
    - Buildings (individual detections with GPS)
    - Permits (cadastral data)
    - Detection metadata (method, parameters)
    """
    
    def __init__(self, db_path: str = "illegal_buildings.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.create_tables()
        logger.info(f"✅ EnhancedDatabaseManager initialized: {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Database error: {e}")
            raise
        finally:
            conn.close()
    
    def create_tables(self):
        """Create all necessary database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # ===== TABLE 1: SESSIONS =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    t1_image_path TEXT NOT NULL,
                    t2_image_path TEXT NOT NULL,
                    image_width INTEGER,
                    image_height INTEGER,
                    change_threshold REAL DEFAULT 30.0,
                    detection_method TEXT DEFAULT 'region_proposal',
                    metadata TEXT
                )
            ''')
            
            # ===== TABLE 2: RESULTS =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    total_buildings INTEGER DEFAULT 0,
                    legal_buildings INTEGER DEFAULT 0,
                    illegal_buildings INTEGER DEFAULT 0,
                    uncertain_buildings INTEGER DEFAULT 0,
                    under_construction INTEGER DEFAULT 0,
                    existing_buildings INTEGER DEFAULT 0,
                    total_area_m2 REAL DEFAULT 0,
                    illegal_area_m2 REAL DEFAULT 0,
                    change_percentage REAL DEFAULT 0,
                    compliance_rate REAL DEFAULT 0,
                    processing_time REAL,
                    statistics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            ''')
            
            # ===== TABLE 3: BUILDINGS (Enhanced with GPS and detection metadata) =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS buildings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    building_id TEXT UNIQUE,
                    center_x INTEGER,
                    center_y INTEGER,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_width INTEGER,
                    bbox_height INTEGER,
                    area_pixels INTEGER,
                    area_m2 REAL,
                    perimeter_m REAL,
                    gps_latitude REAL,
                    gps_longitude REAL,
                    gps_accuracy_m REAL,
                    confidence REAL,
                    change_percentage REAL,
                    status TEXT,
                    legal_status TEXT,
                    legal_reason TEXT,
                    legal_confidence REAL,
                    detection_method TEXT,
                    has_permit BOOLEAN DEFAULT 0,
                    permit_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (permit_id) REFERENCES cadastral_permits(permit_number)
                )
            ''')
            
            # ===== TABLE 4: CADASTRAL PERMITS =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cadastral_permits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    permit_number TEXT UNIQUE,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    address TEXT,
                    owner_name TEXT,
                    permit_date DATE,
                    expiry_date DATE,
                    building_type TEXT,
                    approved_area REAL,
                    status TEXT DEFAULT 'active',
                    geometry TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ===== TABLE 5: DETECTION_METADATA (New - stores detection parameters) =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    gps_resolution_x REAL,
                    gps_resolution_y REAL,
                    coverage_area_km2 REAL,
                    min_building_area INTEGER,
                    nms_threshold REAL,
                    confidence_threshold REAL,
                    scales TEXT,
                    image_bounds TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            ''')
            
            # ===== TABLE 6: GPS_COORDINATES (New - stores building GPS polygons) =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gps_coordinates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    building_id TEXT NOT NULL,
                    corner_index INTEGER,
                    latitude REAL,
                    longitude REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (building_id) REFERENCES buildings(building_id) ON DELETE CASCADE
                )
            ''')
            
            # ===== CREATE INDEXES =====
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_buildings_session ON buildings(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_buildings_legal_status ON buildings(legal_status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_buildings_gps ON buildings(gps_latitude, gps_longitude)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_permits_location ON cadastral_permits(latitude, longitude)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_permits_status ON cadastral_permits(status)')
            
            conn.commit()
            logger.info("✅ All database tables and indexes created successfully")
    
    # ========================================================================
    # SESSION OPERATIONS
    # ========================================================================
    
    def create_session(self, session_id: str, t1_path: str, t2_path: str, 
                       image_size: tuple = None, change_threshold: float = 30.0,
                       detection_method: str = 'region_proposal',
                       metadata: Dict = None) -> str:
        """
        Create a new detection session
        
        Args:
            session_id: Unique session identifier
            t1_path: Path to time 1 image
            t2_path: Path to time 2 image
            image_size: (width, height) tuple
            change_threshold: Change detection threshold
            detection_method: Detection method used
            metadata: Additional metadata dictionary
        
        Returns:
            Session ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            img_width, img_height = image_size if image_size else (None, None)
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute('''
                INSERT INTO sessions (
                    id, t1_image_path, t2_image_path, 
                    image_width, image_height, change_threshold,
                    detection_method, metadata, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'processing')
            ''', (
                session_id, t1_path, t2_path,
                img_width, img_height, change_threshold,
                detection_method, metadata_json
            ))
            
            logger.info(f"✅ Session created: {session_id}")
            return session_id
    
    def update_session_status(self, session_id: str, status: str):
        """Update session status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, session_id))
            logger.info(f"📝 Session {session_id} status: {status}")
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session details"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
            row = cursor.fetchone()
            
            if row:
                session = dict(row)
                if session.get('metadata'):
                    session['metadata'] = json.loads(session['metadata'])
                return session
            return None
    
    # ========================================================================
    # RESULTS OPERATIONS
    # ========================================================================
    
    def save_results(self, session_id: str, buildings: List[Dict], 
                    statistics: Dict, processing_time: float = None):
        """
        Save detection results
        
        Args:
            session_id: Session identifier
            buildings: List of detected buildings
            statistics: Statistics dictionary
            processing_time: Processing time in seconds
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Extract summary statistics
            total_buildings = len(buildings)
            legal_count = len([b for b in buildings if b.get('legal_status') == 'LEGAL'])
            illegal_count = len([b for b in buildings if b.get('legal_status') == 'ILLEGAL'])
            uncertain_count = len([b for b in buildings if b.get('legal_status') not in ['LEGAL', 'ILLEGAL']])
            under_construction = len([b for b in buildings if b.get('status') == 'UNDER_CONSTRUCTION'])
            existing = len([b for b in buildings if b.get('status') == 'EXISTING'])
            
            # Save to results table
            cursor.execute('''
                INSERT INTO results (
                    session_id, total_buildings, legal_buildings, illegal_buildings,
                    uncertain_buildings, under_construction, existing_buildings,
                    total_area_m2, illegal_area_m2, change_percentage, compliance_rate,
                    processing_time, statistics
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, total_buildings, legal_count, illegal_count,
                uncertain_count, under_construction, existing,
                statistics.get('total_building_area_m2', 0),
                statistics.get('illegal_building_area_m2', 0),
                statistics.get('change_percentage', 0),
                statistics.get('compliance_rate', 0),
                processing_time,
                json.dumps(statistics)
            ))
            
            # Save individual buildings
            for building in buildings:
                self._save_building(cursor, session_id, building)
            
            conn.commit()
            logger.info(f"✅ Results saved: {session_id} ({total_buildings} buildings)")
    
    def _save_building(self, cursor, session_id: str, building: Dict):
        """Save individual building to database"""
        bbox = building.get('bbox', {})
        
        cursor.execute('''
            INSERT INTO buildings (
                session_id, building_id, center_x, center_y,
                bbox_x, bbox_y, bbox_width, bbox_height,
                area_pixels, area_m2, perimeter_m,
                gps_latitude, gps_longitude, gps_accuracy_m,
                confidence, change_percentage, status, legal_status,
                legal_reason, legal_confidence, detection_method,
                has_permit, permit_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            building.get('id'),
            building.get('center_x'),
            building.get('center_y'),
            bbox.get('x'),
            bbox.get('y'),
            bbox.get('width'),
            bbox.get('height'),
            building.get('area_pixels'),
            building.get('area_m2'),
            building.get('perimeter_m'),
            building.get('gps_latitude'),
            building.get('gps_longitude'),
            building.get('gps_accuracy_estimate_m'),
            building.get('confidence'),
            building.get('change_percentage'),
            building.get('status'),
            building.get('legal_status'),
            building.get('legal_reason'),
            building.get('legal_confidence'),
            building.get('detection_method'),
            1 if building.get('legal_status') == 'LEGAL' else 0,
            building.get('permit_id')
        ))
        
        # Save GPS polygon corners if available
        if 'gps_corners' in building:
            building_id = building.get('id')
            for idx, corner in enumerate(building['gps_corners']):
                cursor.execute('''
                    INSERT INTO gps_coordinates (building_id, corner_index, latitude, longitude)
                    VALUES (?, ?, ?, ?)
                ''', (building_id, idx, corner.get('lat'), corner.get('lon')))
    
    def save_detection_metadata(self, session_id: str, gps_stats: Dict, 
                               detector_config: Dict):
        """
        Save detection metadata (GPS stats, detector parameters)
        
        Args:
            session_id: Session identifier
            gps_stats: GPS calculator statistics
            detector_config: Detector configuration
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            resolution = gps_stats.get('resolution', {})
            image_bounds = gps_stats.get('bounds', {})
            
            cursor.execute('''
                INSERT INTO detection_metadata (
                    session_id, gps_resolution_x, gps_resolution_y, coverage_area_km2,
                    min_building_area, nms_threshold, confidence_threshold,
                    scales, image_bounds
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                resolution.get('meters_per_pixel_x'),
                resolution.get('meters_per_pixel_y'),
                gps_stats.get('coverage_area_km2'),
                detector_config.get('min_area'),
                detector_config.get('nms_threshold'),
                detector_config.get('confidence_threshold'),
                json.dumps(detector_config.get('scales', [])),
                json.dumps(image_bounds)
            ))
            
            logger.info(f"✅ Detection metadata saved for session: {session_id}")
    
    def get_results(self, session_id: str) -> Optional[Dict]:
        """Get complete results for a session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get session info
            cursor.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
            session_row = cursor.fetchone()
            
            if not session_row:
                return None
            
            session = dict(session_row)
            if session.get('metadata'):
                session['metadata'] = json.loads(session['metadata'])
            
            # Get results summary
            cursor.execute('''
                SELECT * FROM results WHERE session_id = ? 
                ORDER BY created_at DESC LIMIT 1
            ''', (session_id,))
            result_row = cursor.fetchone()
            
            if result_row:
                result = dict(result_row)
                statistics = json.loads(result['statistics']) if result.get('statistics') else {}
            else:
                result = {}
                statistics = {}
            
            # Get buildings
            cursor.execute('SELECT * FROM buildings WHERE session_id = ?', (session_id,))
            buildings = [dict(row) for row in cursor.fetchall()]
            
            # Get GPS corners for each building
            for building in buildings:
                building_id = building['building_id']
                cursor.execute('''
                    SELECT * FROM gps_coordinates 
                    WHERE building_id = ? 
                    ORDER BY corner_index
                ''', (building_id,))
                corners = [
                    {'lat': row['latitude'], 'lon': row['longitude']} 
                    for row in cursor.fetchall()
                ]
                if corners:
                    building['gps_corners'] = corners
            
            # Get detection metadata
            cursor.execute('''
                SELECT * FROM detection_metadata 
                WHERE session_id = ?
            ''', (session_id,))
            metadata_row = cursor.fetchone()
            detection_metadata = dict(metadata_row) if metadata_row else {}
            
            return {
                'session': session,
                'buildings': buildings,
                'statistics': statistics,
                'processing_time': result.get('processing_time'),
                'detection_metadata': detection_metadata
            }
    
    # ========================================================================
    # PERMIT OPERATIONS
    # ========================================================================
    
    def add_permit(self, permit_number: str, latitude: float, longitude: float,
                   **kwargs):
        """Add a cadastral permit"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO cadastral_permits (
                    permit_number, latitude, longitude, address, owner_name,
                    permit_date, expiry_date, building_type, approved_area, 
                    status, geometry
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                permit_number, latitude, longitude,
                kwargs.get('address'),
                kwargs.get('owner_name'),
                kwargs.get('permit_date'),
                kwargs.get('expiry_date'),
                kwargs.get('building_type'),
                kwargs.get('approved_area'),
                kwargs.get('status', 'active'),
                kwargs.get('geometry')
            ))
            
            logger.info(f"✅ Permit added: {permit_number}")
    
    def get_permits_near_location(self, latitude: float, longitude: float,
                                  radius_deg: float = 0.001) -> List[Dict]:
        """Find permits near a location"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM cadastral_permits
                WHERE latitude BETWEEN ? AND ?
                AND longitude BETWEEN ? AND ?
                AND status = 'active'
            ''', (
                latitude - radius_deg, latitude + radius_deg,
                longitude - radius_deg, longitude + radius_deg
            ))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # ========================================================================
    # STATISTICS & QUERIES
    # ========================================================================
    
    def get_overall_statistics(self) -> Dict:
        """Get overall system statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total sessions
            cursor.execute('SELECT COUNT(*) as count FROM sessions')
            stats['total_sessions'] = cursor.fetchone()['count']
            
            # Sessions by status
            cursor.execute('SELECT status, COUNT(*) as count FROM sessions GROUP BY status')
            stats['sessions_by_status'] = {row['status']: row['count'] 
                                          for row in cursor.fetchall()}
            
            # Total buildings
            cursor.execute('SELECT COUNT(*) as count FROM buildings')
            stats['total_buildings'] = cursor.fetchone()['count']
            
            # Buildings by legal status
            cursor.execute('''
                SELECT legal_status, COUNT(*) as count 
                FROM buildings 
                GROUP BY legal_status
            ''')
            stats['buildings_by_legal_status'] = {row['legal_status']: row['count'] 
                                                   for row in cursor.fetchall()}
            
            # Total area
            cursor.execute('SELECT SUM(area_m2) as total FROM buildings')
            stats['total_building_area_m2'] = cursor.fetchone()['total'] or 0
            
            # Illegal area
            cursor.execute('''
                SELECT SUM(area_m2) as total 
                FROM buildings 
                WHERE legal_status = 'ILLEGAL'
            ''')
            stats['illegal_building_area_m2'] = cursor.fetchone()['total'] or 0
            
            # Total permits
            cursor.execute('SELECT COUNT(*) as count FROM cadastral_permits')
            stats['total_permits'] = cursor.fetchone()['count']
            
            # Average compliance rate
            cursor.execute('SELECT AVG(compliance_rate) as avg FROM results')
            stats['average_compliance_rate'] = cursor.fetchone()['avg'] or 0
            
            return stats
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """Get recent sessions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.*, r.total_buildings, r.illegal_buildings, r.compliance_rate
                FROM sessions s
                LEFT JOIN results r ON s.id = r.session_id
                ORDER BY s.created_at DESC
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_session(self, session_id: str):
        """Delete a session and all associated data (CASCADE)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
            logger.info(f"🗑️  Session deleted: {session_id}")
    
    def cleanup_old_sessions(self, days: int = 30) -> int:
        """Clean up sessions older than specified days"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM sessions 
                WHERE created_at < DATE('now', '-' || ? || ' days')
            ''', (days,))
            
            deleted = cursor.rowcount
            logger.info(f"🗑️  Cleaned up {deleted} old sessions")
            return deleted


# Convenience function to replace old DatabaseManager
def get_database_manager(db_path: str = "illegal_buildings.db") -> EnhancedDatabaseManager:
    """Get database manager instance"""
    return EnhancedDatabaseManager(db_path)


if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced Database Manager - Test")
    print("=" * 70)
    
    # Test database creation
    db = EnhancedDatabaseManager("test_db.db")
    
    # Test session creation
    session_id = "test_session_001"
    db.create_session(
        session_id=session_id,
        t1_path="/path/to/img1.png",
        t2_path="/path/to/img2.png",
        image_size=(2048, 2048),
        change_threshold=30.0,
        metadata={'test': True}
    )
    
    print(f"✅ Session created: {session_id}")
    
    # Test building save
    test_building = {
        'id': 'BLDG_TEST001',
        'center_x': 1024,
        'center_y': 1024,
        'bbox': {'x': 1000, 'y': 1000, 'width': 48, 'height': 48},
        'area_pixels': 2304,
        'area_m2': 144.0,
        'perimeter_m': 48.0,
        'gps_latitude': 12.3000,
        'gps_longitude': 76.6500,
        'gps_accuracy_estimate_m': 5.0,
        'confidence': 0.85,
        'change_percentage': 45.0,
        'status': 'UNDER_CONSTRUCTION',
        'legal_status': 'ILLEGAL',
        'legal_reason': 'No permit found',
        'detection_method': 'region_proposal'
    }
    
    statistics = {
        'total_buildings': 1,
        'illegal_buildings': 1,
        'change_percentage': 15.5,
        'compliance_rate': 0.0
    }
    
    db.save_results(session_id, [test_building], statistics, processing_time=5.2)
    print("✅ Results saved")
    
    # Test retrieval
    results = db.get_results(session_id)
    print(f"✅ Retrieved results: {len(results['buildings'])} buildings")
    
    # Test statistics
    stats = db.get_overall_statistics()
    print(f"✅ Overall stats: {stats}")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)