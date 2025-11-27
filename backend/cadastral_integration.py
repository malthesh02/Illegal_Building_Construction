"""
Enhanced Cadastral Map Integration with Permit Database
Connects building detections with cadastral permits and boundaries

Author: Enhanced Version
Date: 2025
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import GIS libraries
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, shape
    from shapely import wkt
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    logger.warning("GeoPandas not available. Limited GIS functionality.")


class CadastralMapIntegration:
    """
    Integrates building detections with cadastral maps and permit database
    """
    
    def __init__(self, db_path: str):
        """Initialize cadastral integration"""
        self.db_path = db_path
        self._ensure_tables()
        logger.info("✅ CadastralMapIntegration initialized")
    
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
    
    def _ensure_tables(self):
        """Ensure cadastral tables exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Cadastral parcels table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cadastral_parcels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parcel_id TEXT UNIQUE NOT NULL,
                    parcel_number TEXT,
                    owner_name TEXT,
                    land_use TEXT,
                    zone_type TEXT,
                    area_m2 REAL,
                    geometry TEXT,
                    boundary_points TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Building-Permit mapping table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS building_permit_mapping (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    building_id TEXT NOT NULL,
                    permit_number TEXT NOT NULL,
                    parcel_id TEXT,
                    overlap_percentage REAL,
                    match_confidence REAL,
                    match_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (building_id) REFERENCES buildings(building_id),
                    FOREIGN KEY (permit_number) REFERENCES cadastral_permits(permit_number),
                    FOREIGN KEY (parcel_id) REFERENCES cadastral_parcels(parcel_id)
                )
            ''')
            
            # Cadastral zones table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cadastral_zones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    zone_id TEXT UNIQUE NOT NULL,
                    zone_name TEXT,
                    zone_type TEXT,
                    regulations TEXT,
                    max_height_m REAL,
                    max_coverage_percentage REAL,
                    geometry TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_parcels_parcel_id ON cadastral_parcels(parcel_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_building_permit_building ON building_permit_mapping(building_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_building_permit_permit ON building_permit_mapping(permit_number)')
            
            conn.commit()
            logger.info("✅ Cadastral tables ensured")
    
    # ========================================================================
    # IMPORT CADASTRAL DATA
    # ========================================================================
    
    def import_cadastral_shapefile(self, shapefile_path: str, 
                                   layer_type: str = 'permits') -> Dict:
        """
        Import cadastral data from shapefile
        
        Args:
            shapefile_path: Path to .shp file
            layer_type: 'permits', 'parcels', or 'zones'
        
        Returns:
            Import statistics
        """
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("GeoPandas is required for shapefile import. Install with: pip install geopandas")
        
        logger.info(f"📂 Importing cadastral shapefile: {shapefile_path}")
        
        try:
            # Read shapefile
            gdf = gpd.read_file(shapefile_path)
            
            # Ensure WGS84 projection
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            
            logger.info(f"Loaded {len(gdf)} features from shapefile")
            
            # Import based on layer type
            if layer_type == 'permits':
                return self._import_permit_layer(gdf)
            elif layer_type == 'parcels':
                return self._import_parcel_layer(gdf)
            elif layer_type == 'zones':
                return self._import_zone_layer(gdf)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        
        except Exception as e:
            logger.error(f"❌ Shapefile import failed: {e}")
            raise
    
    def _import_permit_layer(self, gdf: 'gpd.GeoDataFrame') -> Dict:
        """Import permit layer from GeoDataFrame"""
        imported = 0
        failed = 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for idx, row in gdf.iterrows():
                try:
                    # Extract geometry
                    geom = row.geometry
                    if geom.is_empty:
                        failed += 1
                        continue
                    
                    # Get centroid for lat/lon
                    centroid = geom.centroid
                    
                    # Extract attributes (adjust field names as needed)
                    permit_number = row.get('PERMIT_NO') or row.get('permit_id') or f"AUTO_{idx}"
                    address = row.get('ADDRESS') or row.get('address') or ""
                    owner_name = row.get('OWNER') or row.get('owner_name') or ""
                    building_type = row.get('TYPE') or row.get('building_type') or "Unknown"
                    approved_area = row.get('AREA') or row.get('approved_area') or 0
                    
                    # Insert permit
                    cursor.execute('''
                        INSERT OR REPLACE INTO cadastral_permits (
                            permit_number, latitude, longitude, address, owner_name,
                            building_type, approved_area, geometry, status
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
                    ''', (
                        permit_number,
                        centroid.y,
                        centroid.x,
                        address,
                        owner_name,
                        building_type,
                        approved_area,
                        geom.wkt
                    ))
                    
                    imported += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to import permit {idx}: {e}")
                    failed += 1
            
            conn.commit()
        
        logger.info(f"✅ Imported {imported} permits, {failed} failed")
        
        return {
            'imported': imported,
            'failed': failed,
            'total': len(gdf)
        }
    
    def _import_parcel_layer(self, gdf: 'gpd.GeoDataFrame') -> Dict:
        """Import parcel layer from GeoDataFrame"""
        imported = 0
        failed = 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for idx, row in gdf.iterrows():
                try:
                    geom = row.geometry
                    if geom.is_empty:
                        failed += 1
                        continue
                    
                    # Extract attributes
                    parcel_id = row.get('PARCEL_ID') or f"PARCEL_{idx}"
                    parcel_number = row.get('PARCEL_NO') or ""
                    owner_name = row.get('OWNER') or ""
                    land_use = row.get('LAND_USE') or ""
                    zone_type = row.get('ZONE') or ""
                    area_m2 = geom.area * 111000 * 111000  # Rough conversion
                    
                    # Extract boundary points
                    if hasattr(geom, 'exterior'):
                        coords = list(geom.exterior.coords)
                        boundary_points = json.dumps([{'lat': c[1], 'lon': c[0]} for c in coords])
                    else:
                        boundary_points = None
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO cadastral_parcels (
                            parcel_id, parcel_number, owner_name, land_use,
                            zone_type, area_m2, geometry, boundary_points
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        parcel_id, parcel_number, owner_name, land_use,
                        zone_type, area_m2, geom.wkt, boundary_points
                    ))
                    
                    imported += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to import parcel {idx}: {e}")
                    failed += 1
            
            conn.commit()
        
        logger.info(f"✅ Imported {imported} parcels, {failed} failed")
        
        return {
            'imported': imported,
            'failed': failed,
            'total': len(gdf)
        }
    
    def _import_zone_layer(self, gdf: 'gpd.GeoDataFrame') -> Dict:
        """Import zoning layer from GeoDataFrame"""
        imported = 0
        failed = 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for idx, row in gdf.iterrows():
                try:
                    geom = row.geometry
                    if geom.is_empty:
                        failed += 1
                        continue
                    
                    zone_id = row.get('ZONE_ID') or f"ZONE_{idx}"
                    zone_name = row.get('ZONE_NAME') or ""
                    zone_type = row.get('ZONE_TYPE') or ""
                    max_height = row.get('MAX_HEIGHT') or None
                    max_coverage = row.get('MAX_COVER') or None
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO cadastral_zones (
                            zone_id, zone_name, zone_type, max_height_m,
                            max_coverage_percentage, geometry
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        zone_id, zone_name, zone_type, max_height,
                        max_coverage, geom.wkt
                    ))
                    
                    imported += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to import zone {idx}: {e}")
                    failed += 1
            
            conn.commit()
        
        logger.info(f"✅ Imported {imported} zones, {failed} failed")
        
        return {
            'imported': imported,
            'failed': failed,
            'total': len(gdf)
        }
    
    # ========================================================================
    # BUILDING-PERMIT MATCHING
    # ========================================================================
    
    def match_building_to_permits(self, building: Dict) -> Dict:
        """
        Match a building to cadastral permits
        
        Args:
            building: Building dictionary with GPS coordinates
        
        Returns:
            Match result with permit info
        """
        lat = building.get('gps_latitude')
        lon = building.get('gps_longitude')
        
        if not lat or not lon:
            return {
                'matched': False,
                'reason': 'No GPS coordinates available',
                'permits': []
            }
        
        # Get nearby permits
        permits = self._find_permits_near_point(lat, lon, radius_m=100)
        
        if not permits:
            return {
                'matched': False,
                'reason': 'No permits found within 100m radius',
                'permits': []
            }
        
        # Check for geometric overlap if polygons available
        if 'gps_corners' in building and GEOPANDAS_AVAILABLE:
            matches = self._check_polygon_overlap(building, permits)
        else:
            # Use simple distance-based matching
            matches = self._check_distance_match(lat, lon, permits)
        
        if matches:
            best_match = max(matches, key=lambda m: m['confidence'])
            
            # Save mapping to database
            self._save_building_permit_mapping(
                building_id=building.get('id'),
                permit_number=best_match['permit_number'],
                overlap_percentage=best_match.get('overlap', 0),
                match_confidence=best_match['confidence'],
                match_type=best_match['type']
            )
            
            return {
                'matched': True,
                'best_match': best_match,
                'all_matches': matches,
                'permits': permits
            }
        
        return {
            'matched': False,
            'reason': 'No matching permits found',
            'permits': permits
        }
    
    def _find_permits_near_point(self, lat: float, lon: float, 
                                 radius_m: float = 100) -> List[Dict]:
        """Find permits near a point"""
        # Convert radius to degrees (approximate)
        radius_deg = radius_m / 111000
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM cadastral_permits
                WHERE latitude BETWEEN ? AND ?
                AND longitude BETWEEN ? AND ?
                AND status = 'active'
            ''', (
                lat - radius_deg, lat + radius_deg,
                lon - radius_deg, lon + radius_deg
            ))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def _check_polygon_overlap(self, building: Dict, permits: List[Dict]) -> List[Dict]:
        """Check polygon overlap between building and permits"""
        if not GEOPANDAS_AVAILABLE:
            return []
        
        matches = []
        
        try:
            # Create building polygon
            corners = building.get('gps_corners', [])
            if len(corners) < 3:
                return []
            
            building_poly = Polygon([(c['lon'], c['lat']) for c in corners])
            
            for permit in permits:
                if not permit.get('geometry'):
                    continue
                
                try:
                    permit_geom = wkt.loads(permit['geometry'])
                    
                    # Check overlap
                    if permit_geom.intersects(building_poly):
                        intersection = permit_geom.intersection(building_poly)
                        overlap = (intersection.area / building_poly.area) * 100
                        
                        # Calculate confidence based on overlap
                        if overlap >= 80:
                            confidence = 0.95
                            match_type = 'full_overlap'
                        elif overlap >= 50:
                            confidence = 0.75
                            match_type = 'partial_overlap'
                        else:
                            confidence = 0.50
                            match_type = 'edge_overlap'
                        
                        matches.append({
                            'permit_number': permit['permit_number'],
                            'overlap': round(overlap, 2),
                            'confidence': confidence,
                            'type': match_type,
                            'permit': permit
                        })
                
                except Exception as e:
                    logger.warning(f"Error checking overlap for permit {permit.get('permit_number')}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in polygon overlap check: {e}")
        
        return matches
    
    def _check_distance_match(self, lat: float, lon: float, 
                             permits: List[Dict]) -> List[Dict]:
        """Simple distance-based matching"""
        matches = []
        
        for permit in permits:
            p_lat = permit.get('latitude')
            p_lon = permit.get('longitude')
            
            if not p_lat or not p_lon:
                continue
            
            # Calculate distance
            distance_m = self._haversine_distance(lat, lon, p_lat, p_lon)
            
            # Calculate confidence based on distance
            if distance_m <= 10:
                confidence = 0.90
                match_type = 'very_close'
            elif distance_m <= 25:
                confidence = 0.75
                match_type = 'close'
            elif distance_m <= 50:
                confidence = 0.60
                match_type = 'nearby'
            else:
                confidence = 0.40
                match_type = 'distant'
            
            matches.append({
                'permit_number': permit['permit_number'],
                'distance_m': round(distance_m, 2),
                'confidence': confidence,
                'type': match_type,
                'permit': permit
            })
        
        return matches
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS points in meters"""
        import math
        
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        
        a = (math.sin(dphi / 2) ** 2 + 
             math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _save_building_permit_mapping(self, building_id: str, permit_number: str,
                                     overlap_percentage: float, match_confidence: float,
                                     match_type: str, parcel_id: str = None):
        """Save building-permit mapping"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO building_permit_mapping (
                    building_id, permit_number, parcel_id, overlap_percentage,
                    match_confidence, match_type
                )
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                building_id, permit_number, parcel_id, overlap_percentage,
                match_confidence, match_type
            ))
    
    # ========================================================================
    # BATCH OPERATIONS
    # ========================================================================
    
    def batch_match_buildings(self, buildings: List[Dict]) -> Dict:
        """
        Match multiple buildings to permits in batch
        
        Args:
            buildings: List of building dictionaries
        
        Returns:
            Batch matching statistics
        """
        logger.info(f"🔄 Batch matching {len(buildings)} buildings to permits...")
        
        results = {
            'total': len(buildings),
            'matched': 0,
            'unmatched': 0,
            'matches': []
        }
        
        for building in buildings:
            match_result = self.match_building_to_permits(building)
            
            if match_result['matched']:
                results['matched'] += 1
            else:
                results['unmatched'] += 1
            
            results['matches'].append({
                'building_id': building.get('id'),
                'matched': match_result['matched'],
                'best_match': match_result.get('best_match'),
                'reason': match_result.get('reason')
            })
        
        logger.info(f"✅ Batch matching complete: {results['matched']}/{results['total']} matched")
        
        return results
    
    # ========================================================================
    # EXPORT & VISUALIZATION
    # ========================================================================
    
    def export_cadastral_geojson(self, output_path: str, 
                                 include_permits: bool = True,
                                 include_parcels: bool = False) -> str:
        """Export cadastral data to GeoJSON"""
        features = []
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if include_permits:
                cursor.execute('SELECT * FROM cadastral_permits WHERE status = "active"')
                for row in cursor.fetchall():
                    permit = dict(row)
                    if permit.get('geometry'):
                        try:
                            geom = wkt.loads(permit['geometry'])
                            features.append({
                                'type': 'Feature',
                                'geometry': json.loads(gpd.GeoSeries([geom]).to_json())['features'][0]['geometry'] if GEOPANDAS_AVAILABLE else None,
                                'properties': {
                                    'type': 'permit',
                                    'permit_number': permit['permit_number'],
                                    'owner_name': permit.get('owner_name'),
                                    'approved_area': permit.get('approved_area')
                                }
                            })
                        except:
                            pass
            
            if include_parcels:
                cursor.execute('SELECT * FROM cadastral_parcels')
                for row in cursor.fetchall():
                    parcel = dict(row)
                    if parcel.get('geometry'):
                        try:
                            geom = wkt.loads(parcel['geometry'])
                            features.append({
                                'type': 'Feature',
                                'geometry': json.loads(gpd.GeoSeries([geom]).to_json())['features'][0]['geometry'] if GEOPANDAS_AVAILABLE else None,
                                'properties': {
                                    'type': 'parcel',
                                    'parcel_id': parcel['parcel_id'],
                                    'owner_name': parcel.get('owner_name'),
                                    'land_use': parcel.get('land_use')
                                }
                            })
                        except:
                            pass
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        logger.info(f"✅ Cadastral GeoJSON exported: {output_path} ({len(features)} features)")
        return output_path
    
    def get_statistics(self) -> Dict:
        """Get cadastral database statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            cursor.execute('SELECT COUNT(*) as count FROM cadastral_permits WHERE status = "active"')
            stats['active_permits'] = cursor.fetchone()['count']
            
            cursor.execute('SELECT COUNT(*) as count FROM cadastral_parcels')
            stats['total_parcels'] = cursor.fetchone()['count']
            
            cursor.execute('SELECT COUNT(*) as count FROM cadastral_zones')
            stats['total_zones'] = cursor.fetchone()['count']
            
            cursor.execute('SELECT COUNT(*) as count FROM building_permit_mapping')
            stats['mapped_buildings'] = cursor.fetchone()['count']
            
            cursor.execute('''
                SELECT match_type, COUNT(*) as count 
                FROM building_permit_mapping 
                GROUP BY match_type
            ''')
            stats['matches_by_type'] = {row['match_type']: row['count'] 
                                        for row in cursor.fetchall()}
            
            return stats


# Test code
if __name__ == "__main__":
    print("=" * 70)
    print("Cadastral Map Integration - Test")
    print("=" * 70)
    
    cadastral = CadastralMapIntegration("test_cadastral.db")
    
    # Test building matching
    test_building = {
        'id': 'BLDG_TEST001',
        'gps_latitude': 12.3000,
        'gps_longitude': 76.6500,
        'area_m2': 144.0
    }
    
    match_result = cadastral.match_building_to_permits(test_building)
    print(f"Match result: {match_result}")
    
    # Get statistics
    stats = cadastral.get_statistics()
    print(f"Statistics: {stats}")
    
    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)