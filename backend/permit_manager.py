import os
import math
import json
import logging
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Polygon, Point, box
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PermitManager:
    """
    Comprehensive Permit Manager with GIS-accurate overlap detection,
    GeoJSON export/import, and advanced spatial analysis capabilities.
    """

    def __init__(self, db_path: str, image_bounds=None):
        """
        Initialize PermitManager with database connection and bounds.
        
        Args:
            db_path: Path to SQLite database
            image_bounds: Optional custom GPS boundaries
        """
        from database_manager import EnhancedDatabaseManager
        self.db_path = db_path
        self.db = EnhancedDatabaseManager(db_path)
        self.image_bounds = image_bounds or self._get_default_bounds()
        self._ensure_db_structure()
        logger.info("✅ PermitManager initialized successfully")

    # =====================================================================
    # DEFAULT MAP BOUNDS
    # =====================================================================
    def _get_default_bounds(self) -> Dict[str, float]:
        """
        Get default GPS bounds for Mysuru region.
        
        Returns:
            Dict with north, south, east, west GPS coordinates
        """
        return {
            'north': 12.3200,
            'south': 12.2900,
            'east': 76.6600,
            'west': 76.6300
        }

    # =====================================================================
    # DATABASE SETUP & MANAGEMENT
    # =====================================================================
    def _ensure_db_structure(self):
        """Create cadastral_permits table if it doesn't exist."""
        with self.db.get_connection() as conn:
            c = conn.cursor()
            
            # Create main cadastral permits table
            c.execute("""
                CREATE TABLE IF NOT EXISTS cadastral_permits (
                    permit_number TEXT PRIMARY KEY,
                    address TEXT,
                    owner_name TEXT,
                    building_type TEXT,
                    permit_date TEXT,
                    approved_area REAL,
                    status TEXT DEFAULT 'active',
                    geometry TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for faster queries
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_permit_status 
                ON cadastral_permits(status)
            """)
            
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_permit_number 
                ON cadastral_permits(permit_number)
            """)
            
            # Create audit log table
            c.execute("""
                CREATE TABLE IF NOT EXISTS import_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT,
                    file_size_bytes INTEGER,
                    feature_count INTEGER,
                    successful_imports INTEGER,
                    failed_imports INTEGER,
                    import_duration_seconds REAL,
                    status TEXT
                )
            """)
            
            conn.commit()
            logger.info("✅ Database structure ready (cadastral_permits + audit tables)")

    def get_db_statistics(self) -> Dict:
        """
        Get detailed database statistics.
        
        Returns:
            Dict with table sizes, counts, and metadata
        """
        with self.db.get_connection() as conn:
            c = conn.cursor()
            
            # Count active permits
            c.execute("SELECT COUNT(*) as count FROM cadastral_permits WHERE status='active'")
            active_count = c.fetchone()['count']
            
            # Count inactive permits
            c.execute("SELECT COUNT(*) as count FROM cadastral_permits WHERE status='inactive'")
            inactive_count = c.fetchone()['count']
            
            # Total area covered
            c.execute("SELECT SUM(approved_area) as total_area FROM cadastral_permits WHERE status='active'")
            total_area = c.fetchone()['total_area'] or 0
            
            # Get latest import
            c.execute("""
                SELECT import_date, file_path, feature_count, successful_imports 
                FROM import_audit 
                ORDER BY import_date DESC 
                LIMIT 1
            """)
            latest_import = c.fetchone()
            
            return {
                'active_permits': active_count,
                'inactive_permits': inactive_count,
                'total_permits': active_count + inactive_count,
                'total_area_m2': round(total_area, 2),
                'latest_import': dict(latest_import) if latest_import else None,
                'coordinate_system': 'GPS (WGS84)',
                'default_bounds': self.image_bounds
            }

    # =====================================================================
    # FILE IMPORT (CSV / GeoJSON / Shapefile)
    # =====================================================================
    def import_permits(self, file_path: str) -> Dict:
        """
        Import permits from various file formats.
        
        Supports: CSV, GeoJSON, JSON, Shapefile
        
        Args:
            file_path: Path to the file to import
            
        Returns:
            Dict with import statistics
        """
        import time
        start_time = time.time()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        
        logger.info(f"📂 Starting import from: {file_path}")
        logger.info(f"   File size: {file_size / 1024:.2f} KB")
        
        try:
            # Load data based on file type
            if ext == ".csv":
                df = pd.read_csv(file_path)
                logger.info(f"   Loaded CSV with {len(df)} rows")
                
            elif ext in (".geojson", ".json"):
                gdf = gpd.read_file(file_path).to_crs(epsg=4326)
                df = pd.DataFrame(gdf)
                if "geometry" in df.columns:
                    df["geometry"] = df["geometry"].apply(
                        lambda g: g.wkt if g and not g.is_empty else None
                    )
                logger.info(f"   Loaded GeoJSON with {len(df)} features")
                
            elif ext == ".shp":
                gdf = gpd.read_file(file_path).to_crs(epsg=4326)
                df = pd.DataFrame(gdf)
                if "geometry" in df.columns:
                    df["geometry"] = df["geometry"].apply(
                        lambda g: g.wkt if g and not g.is_empty else None
                    )
                logger.info(f"   Loaded Shapefile with {len(df)} features")
                
            else:
                raise ValueError(f"Unsupported file type: {ext}. Use CSV, GeoJSON, JSON, or SHP")
            
            # Import data into database
            inserted, failed = self._insert_permits_to_db(df)
            
            duration = time.time() - start_time
            
            # Log to audit table
            self._log_import_audit(
                file_path=file_path,
                file_size=file_size,
                feature_count=len(df),
                successful=inserted,
                failed=failed,
                duration=duration
            )
            
            logger.info(f"✅ Import completed in {duration:.2f}s")
            logger.info(f"   Inserted: {inserted}, Failed: {failed}")
            
            return {
                'file_path': file_path,
                'total_records': len(df),
                'inserted': inserted,
                'failed': failed,
                'duration_seconds': round(duration, 2),
                'success_rate': round((inserted / len(df) * 100) if len(df) > 0 else 0, 2)
            }
            
        except Exception as e:
            logger.error(f"❌ Import failed: {e}", exc_info=True)
            self._log_import_audit(
                file_path=file_path,
                file_size=file_size,
                feature_count=0,
                successful=0,
                failed=1,
                duration=time.time() - start_time,
                status='FAILED'
            )
            raise

    def _insert_permits_to_db(self, df: pd.DataFrame) -> Tuple[int, int]:
        """
        Insert permit records into database.
        
        Args:
            df: DataFrame with permit data
            
        Returns:
            Tuple of (inserted_count, failed_count)
        """
        with self.db.get_connection() as conn:
            c = conn.cursor()
            inserted = 0
            failed = 0
            
            for idx, row in df.iterrows():
                try:
                    # Extract data with defaults
                    permit_number = row.get("permit_number") or row.get("id") or f"AUTO_{idx}"
                    address = str(row.get("address", "")).strip() or ""
                    owner_name = str(row.get("owner_name", "")).strip() or ""
                    building_type = str(row.get("building_type", "")).strip() or "Unknown"
                    permit_date = str(row.get("permit_date", "")).strip() or datetime.now().isoformat()
                    approved_area = float(row.get("approved_area", 0)) if row.get("approved_area") else 0
                    geometry = row.get("geometry", None)
                    
                    # Validate geometry if present
                    if geometry:
                        try:
                            wkt.loads(geometry)
                        except Exception as e:
                            logger.warning(f"Invalid geometry for {permit_number}: {e}")
                            geometry = None
                    
                    c.execute("""
                        INSERT OR REPLACE INTO cadastral_permits
                        (permit_number, address, owner_name, building_type, 
                         permit_date, approved_area, geometry, status, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 'active', CURRENT_TIMESTAMP)
                    """, (
                        permit_number,
                        address,
                        owner_name,
                        building_type,
                        permit_date,
                        approved_area,
                        geometry
                    ))
                    inserted += 1
                    
                except Exception as e:
                    logger.warning(f"Row {idx}: {e}")
                    failed += 1
            
            conn.commit()
        
        return inserted, failed

    def _log_import_audit(self, file_path: str, file_size: int, feature_count: int,
                         successful: int, failed: int, duration: float, status: str = 'SUCCESS'):
        """Log import operation to audit table."""
        with self.db.get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO import_audit
                (file_path, file_size_bytes, feature_count, successful_imports, 
                 failed_imports, import_duration_seconds, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (file_path, file_size, feature_count, successful, failed, duration, status))
            conn.commit()

    # =====================================================================
    # COORDINATE CONVERSION UTILITIES
    # =====================================================================
    def pixel_to_gps(self, x: float, y: float, width: int, height: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to GPS coordinates.
        
        Args:
            x, y: Pixel coordinates
            width, height: Image dimensions
            
        Returns:
            Tuple of (latitude, longitude)
        """
        nx = x / width if width > 0 else 0
        ny = y / height if height > 0 else 0
        
        lon = self.image_bounds["west"] + nx * (
            self.image_bounds["east"] - self.image_bounds["west"]
        )
        lat = self.image_bounds["north"] - ny * (
            self.image_bounds["north"] - self.image_bounds["south"]
        )
        
        return lat, lon

    def gps_to_pixel(self, lat: float, lon: float, width: int, height: int) -> Tuple[int, int]:
        """
        Convert GPS coordinates to pixel coordinates.
        
        Args:
            lat, lon: GPS coordinates
            width, height: Image dimensions
            
        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        nx = (lon - self.image_bounds["west"]) / (
            self.image_bounds["east"] - self.image_bounds["west"]
        )
        ny = (self.image_bounds["north"] - lat) / (
            self.image_bounds["north"] - self.image_bounds["south"]
        )
        
        x = int(nx * width)
        y = int(ny * height)
        
        return x, y

    def gps_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance in meters between two GPS points (Haversine formula).
        
        Args:
            lat1, lon1: First point
            lat2, lon2: Second point
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters
        lat1, lat2 = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

    def set_image_bounds(self, north: float, south: float, east: float, west: float):
        """
        Update image GPS boundary box.
        
        Args:
            north, south, east, west: GPS coordinates
        """
        self.image_bounds = {
            'north': north,
            'south': south,
            'east': east,
            'west': west
        }
        logger.info(f"✅ Updated image bounds: {self.image_bounds}")

    # =====================================================================
    # MAIN LEGALITY CHECK (Full + Partial Overlaps)
    # =====================================================================
    def check_building_legality(self, building: Dict, img_w: int, img_h: int) -> Dict:
        """
        Comprehensive building legality check with overlap analysis.
        
        Args:
            building: Dict with center_x, center_y, area_pixels, optional polygon
            img_w, img_h: Image dimensions
            
        Returns:
            Dict with legality status, confidence, and permit info
        """
        # Convert pixel to GPS coordinates
        lat, lon = self.pixel_to_gps(building["center_x"], building["center_y"], img_w, img_h)
        building["gps_latitude"] = lat
        building["gps_longitude"] = lon
        
        # Create building polygon
        if "polygon" in building and building["polygon"]:
            try:
                building_poly = Polygon(building["polygon"])
            except Exception as e:
                logger.warning(f"Invalid polygon, creating buffer: {e}")
                offset = 0.0001
                building_poly = Polygon([
                    (lon - offset, lat - offset),
                    (lon + offset, lat - offset),
                    (lon + offset, lat + offset),
                    (lon - offset, lat + offset)
                ])
        else:
            # Create small buffer around center point
            offset = 0.0001
            building_poly = Polygon([
                (lon - offset, lat - offset),
                (lon + offset, lat - offset),
                (lon + offset, lat + offset),
                (lon - offset, lat + offset)
            ])
        
        # Get all active permits
        with self.db.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM cadastral_permits WHERE status='active'")
            permits = [dict(row) for row in c.fetchall()]
        
        # Calculate estimated area
        pixel_to_meter = 0.5
        area_m2 = building.get("area_pixels", 0) * (pixel_to_meter ** 2)
        building["estimated_area_m2"] = area_m2
        
        # Check against permits
        best_match = None
        best_overlap = 0
        
        for permit in permits:
            if not permit.get("geometry"):
                continue
            
            try:
                permit_geom = wkt.loads(permit["geometry"])
                
                # Check if building is fully inside permit
                if permit_geom.contains(building_poly):
                    return {
                        "status": "LEGAL",
                        "reason": f"Fully inside permitted boundary ({permit['permit_number']})",
                        "confidence": 99,
                        "permit_info": permit,
                        "overlap_ratio": 1.0,
                        "gps_coordinates": {"lat": lat, "lon": lon},
                        "area_m2": area_m2
                    }
                
                # Check for partial overlap
                if permit_geom.intersects(building_poly):
                    try:
                        intersection = permit_geom.intersection(building_poly)
                        inter_area = intersection.area
                        building_area = building_poly.area
                        overlap_ratio = inter_area / building_area if building_area > 0 else 0
                        
                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            best_match = (permit, overlap_ratio)
                    except Exception as e:
                        logger.warning(f"Intersection error: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Geometry parsing error for {permit.get('permit_number')}: {e}")
                continue
        
        # Return best partial match or illegal
        if best_match:
            permit, overlap_ratio = best_match
            
            if overlap_ratio >= 0.8:
                status = "LEGAL"
                confidence = 95
                reason = f"Mostly inside permitted boundary ({overlap_ratio*100:.1f}% overlap)"
            elif overlap_ratio >= 0.6:
                status = "LIKELY LEGAL"
                confidence = 85
                reason = f"Significant overlap with permit ({overlap_ratio*100:.1f}%)"
            elif overlap_ratio >= 0.4:
                status = "UNCERTAIN"
                confidence = 70
                reason = f"Partial overlap - needs field verification ({overlap_ratio*100:.1f}%)"
            elif overlap_ratio >= 0.2:
                status = "LIKELY ILLEGAL"
                confidence = 65
                reason = f"Mostly outside legal boundary ({overlap_ratio*100:.1f}% overlap)"
            else:
                status = "ILLEGAL"
                confidence = 60
                reason = "Building crosses permit boundary but mostly outside"
            
            return {
                "status": status,
                "reason": reason,
                "confidence": confidence,
                "permit_info": permit,
                "overlap_ratio": round(overlap_ratio, 3),
                "gps_coordinates": {"lat": lat, "lon": lon},
                "area_m2": area_m2
            }
        
        # No permit found
        return {
            "status": "ILLEGAL",
            "reason": "No matching cadastral permit found",
            "confidence": 80,
            "permit_info": None,
            "overlap_ratio": 0.0,
            "gps_coordinates": {"lat": lat, "lon": lon},
            "area_m2": area_m2
        }

    def find_permits_near_location(self, lat: float, lon: float, 
                                  radius_meters: float = 500) -> List[Dict]:
        """
        Find all permits within a specified radius of a location.
        
        Args:
            lat, lon: Center GPS coordinates
            radius_meters: Search radius
            
        Returns:
            List of nearby permits with distances
        """
        center_point = Point(lon, lat)
        results = []
        
        with self.db.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM cadastral_permits WHERE status='active'")
            permits = [dict(row) for row in c.fetchall()]
        
        for permit in permits:
            if not permit.get("geometry"):
                continue
            
            try:
                permit_geom = wkt.loads(permit["geometry"])
                distance = center_point.distance(permit_geom) * 111000  # Approximate meters per degree
                
                if distance <= radius_meters:
                    results.append({
                        'permit': permit,
                        'distance_meters': round(distance, 2)
                    })
            except Exception as e:
                logger.warning(f"Error checking distance for {permit.get('permit_number')}: {e}")
        
        return sorted(results, key=lambda x: x['distance_meters'])

    # =====================================================================
    # EXPORT FUNCTIONS
    # =====================================================================
    def export_permits_to_geojson(self, output_path: str) -> str:
        """
        Export all active cadastral permits to GeoJSON format.
        
        Args:
            output_path: Path for output GeoJSON file
            
        Returns:
            Path to created GeoJSON file
        """
        try:
            with self.db.get_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT * FROM cadastral_permits WHERE status='active'")
                permits = [dict(row) for row in c.fetchall()]
            
            geojson = {"type": "FeatureCollection", "features": []}
            
            for permit in permits:
                geom_wkt = permit.get("geometry")
                if not geom_wkt:
                    continue
                
                try:
                    geom = wkt.loads(geom_wkt)
                    if geom.is_empty:
                        continue
                    
                    # Convert geometry to GeoJSON coordinates
                    if geom.geom_type == 'Polygon':
                        coords = [list(geom.exterior.coords)]
                    elif geom.geom_type == 'MultiPolygon':
                        coords = [list(poly.exterior.coords) for poly in geom.geoms]
                    else:
                        continue
                    
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon" if geom.geom_type == 'Polygon' else "MultiPolygon",
                            "coordinates": coords
                        },
                        "properties": {
                            "permit_number": permit.get("permit_number"),
                            "owner_name": permit.get("owner_name"),
                            "address": permit.get("address"),
                            "building_type": permit.get("building_type"),
                            "approved_area": permit.get("approved_area"),
                            "permit_date": permit.get("permit_date"),
                            "status": permit.get("status")
                        }
                    }
                    geojson["features"].append(feature)
                    
                except Exception as e:
                    logger.warning(f"Error processing geometry for {permit.get('permit_number')}: {e}")
                    continue
            
            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(geojson, f, indent=2)
            
            logger.info(f"✅ Exported {len(geojson['features'])} permits to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            # Create empty GeoJSON as fallback
            empty_geojson = {"type": "FeatureCollection", "features": []}
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(empty_geojson, f, indent=2)
            return output_path

    def export_to_csv(self, output_path: str) -> str:
        """
        Export permits to CSV format.
        
        Args:
            output_path: Path for output CSV file
            
        Returns:
            Path to created CSV file
        """
        with self.db.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM cadastral_permits WHERE status='active'")
            permits = [dict(row) for row in c.fetchall()]
        
        df = pd.DataFrame(permits)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"✅ Exported {len(permits)} permits to CSV: {output_path}")
        return output_path

    # =====================================================================
    # QUERY & RETRIEVAL FUNCTIONS
    # =====================================================================
    def get_all_permits(self) -> List[Dict]:
        """Get all active permits."""
        with self.db.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM cadastral_permits WHERE status='active'")
            return [dict(row) for row in c.fetchall()]

    def get_permit_by_number(self, permit_number: str) -> Optional[Dict]:
        """Get a specific permit by permit number."""
        with self.db.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM cadastral_permits WHERE permit_number=?", (permit_number,))
            row = c.fetchone()
            return dict(row) if row else None

    def search_permits(self, query: str) -> List[Dict]:
        """
        Search permits by address, owner name, or building type.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching permits
        """
        search_term = f"%{query}%"
        with self.db.get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT * FROM cadastral_permits 
                WHERE status='active' AND (
                    address LIKE ? OR 
                    owner_name LIKE ? OR 
                    building_type LIKE ?
                )
            """, (search_term, search_term, search_term))
            return [dict(row) for row in c.fetchall()]

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about permits."""
        stats = self.get_db_statistics()
        return stats

    def get_import_history(self, limit: int = 10) -> List[Dict]:
        """Get import audit history."""
        with self.db.get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT * FROM import_audit 
                ORDER BY import_date DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in c.fetchall()]

    # =====================================================================
    # PERMIT MANAGEMENT
    # =====================================================================
    def update_permit_status(self, permit_number: str, status: str) -> bool:
        """
        Update permit status (active/inactive).
        
        Args:
            permit_number: Permit ID
            status: New status
            
        Returns:
            Success boolean
        """
        if status not in ('active', 'inactive'):
            logger.error(f"Invalid status: {status}")
            return False
        
        with self.db.get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                UPDATE cadastral_permits 
                SET status=?, updated_at=CURRENT_TIMESTAMP 
                WHERE permit_number=?
            """, (status, permit_number))
            conn.commit()
        
        logger.info(f"✅ Updated permit {permit_number} status to {status}")
        return True

    def delete_permit(self, permit_number: str) -> bool:
        """Delete a permit from database."""
        with self.db.get_connection() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM cadastral_permits WHERE permit_number=?", (permit_number,))
            conn.commit()
        
        logger.info(f"✅ Deleted permit {permit_number}")
        return True

    def get_permits_by_building_type(self, building_type: str) -> List[Dict]:
        """Get all permits of a specific building type."""
        with self.db.get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT * FROM cadastral_permits 
                WHERE status='active' AND building_type=?
            """, (building_type,))
            return [dict(row) for row in c.fetchall()]

    def get_coverage_summary(self) -> Dict:
        """Get summary of permit coverage in the region."""
        with self.db.get_connection() as conn:
            c = conn.cursor()
            
            c.execute("SELECT COUNT(*) as count FROM cadastral_permits WHERE status='active'")
            total_permits = c.fetchone()['count']
            
            c.execute("SELECT SUM(approved_area) as total FROM cadastral_permits WHERE status='active'")
            total_area = c.fetchone()['total'] or 0
            
            c.execute("""
                SELECT building_type, COUNT(*) as count, SUM(approved_area) as area
                FROM cadastral_permits 
                WHERE status='active'
                GROUP BY building_type
            """)
            by_type = [dict(row) for row in c.fetchall()]
            
            # Calculate bounds coverage
            bounds_area = (
                (self.image_bounds['east'] - self.image_bounds['west']) *
                (self.image_bounds['north'] - self.image_bounds['south']) *
                111000 * 111000  # Rough approximation
            )
            
            return {
                'total_permits': total_permits,
                'total_covered_area_m2': round(total_area, 2),
                'estimated_bounds_area_m2': round(bounds_area, 2),
                'coverage_percentage': round((total_area / bounds_area * 100) if bounds_area > 0 else 0, 2),
                'by_building_type': by_type
            }