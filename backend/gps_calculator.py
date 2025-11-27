
import math
import numpy as np
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class GPSCoordinateCalculator:
    """
    Advanced GPS coordinate calculator with projection handling,
    accuracy improvements, and validation.
    """
    
    def __init__(self, image_bounds: Dict[str, float], image_size: Tuple[int, int]):
        """
        Initialize GPS calculator
        
        Args:
            image_bounds: Dict with 'north', 'south', 'east', 'west' GPS coordinates
            image_size: Tuple of (width, height) in pixels
        """
        self.bounds = image_bounds
        self.width, self.height = image_size
        
        # Calculate image center for projection accuracy
        self.center_lat = (image_bounds['north'] + image_bounds['south']) / 2
        self.center_lon = (image_bounds['east'] + image_bounds['west']) / 2
        
        # Calculate meters per pixel (approximate, improves with real GSD)
        self._calculate_resolution()
        
        logger.info(f"GPS Calculator initialized:")
        logger.info(f"  Image size: {self.width}x{self.height}")
        logger.info(f"  Center: ({self.center_lat:.6f}, {self.center_lon:.6f})")
        logger.info(f"  Resolution: {self.meters_per_pixel_x:.2f}m/px (X), "
                   f"{self.meters_per_pixel_y:.2f}m/px (Y)")
    
    def _calculate_resolution(self):
        """
        Calculate approximate meters per pixel using Haversine distance
        More accurate than simple degree calculations
        """
        # Calculate width in meters at center latitude
        width_meters = self.haversine_distance(
            self.center_lat, self.bounds['west'],
            self.center_lat, self.bounds['east']
        )
        
        # Calculate height in meters
        height_meters = self.haversine_distance(
            self.bounds['north'], self.center_lon,
            self.bounds['south'], self.center_lon
        )
        
        self.meters_per_pixel_x = width_meters / self.width if self.width > 0 else 0
        self.meters_per_pixel_y = height_meters / self.height if self.height > 0 else 0
        
        # Store for area calculations
        self.pixel_area_m2 = self.meters_per_pixel_x * self.meters_per_pixel_y
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate accurate distance between two GPS points using Haversine formula
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def pixel_to_gps(self, x: float, y: float, 
                     method: str = 'bilinear') -> Tuple[float, float]:
        """
        Convert pixel coordinates to GPS coordinates with improved accuracy
        
        Args:
            x, y: Pixel coordinates (0,0 is top-left)
            method: 'simple', 'bilinear', or 'mercator'
            
        Returns:
            Tuple of (latitude, longitude)
        """
        if method == 'simple':
            return self._pixel_to_gps_simple(x, y)
        elif method == 'bilinear':
            return self._pixel_to_gps_bilinear(x, y)
        elif method == 'mercator':
            return self._pixel_to_gps_mercator(x, y)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _pixel_to_gps_simple(self, x: float, y: float) -> Tuple[float, float]:
        """
        Simple linear interpolation (fast but less accurate)
        Use for quick calculations or when high accuracy not needed
        """
        nx = x / self.width if self.width > 0 else 0
        ny = y / self.height if self.height > 0 else 0
        
        lon = self.bounds['west'] + nx * (self.bounds['east'] - self.bounds['west'])
        lat = self.bounds['north'] - ny * (self.bounds['north'] - self.bounds['south'])
        
        return lat, lon
    
    def _pixel_to_gps_bilinear(self, x: float, y: float) -> Tuple[float, float]:
        """
        Bilinear interpolation accounting for latitude distortion
        More accurate than simple linear interpolation
        """
        # Normalize coordinates [0, 1]
        nx = np.clip(x / self.width, 0, 1)
        ny = np.clip(y / self.height, 0, 1)
        
        # Calculate latitude (simple linear)
        lat = self.bounds['north'] - ny * (self.bounds['north'] - self.bounds['south'])
        
        # Calculate longitude with latitude-dependent correction
        # Longitude degrees per pixel varies with latitude
        lat_correction = math.cos(math.radians(lat))
        lon = self.bounds['west'] + nx * (self.bounds['east'] - self.bounds['west']) / lat_correction
        
        # Clamp to bounds
        lon = np.clip(lon, self.bounds['west'], self.bounds['east'])
        lat = np.clip(lat, self.bounds['south'], self.bounds['north'])
        
        return lat, lon
    
    def _pixel_to_gps_mercator(self, x: float, y: float) -> Tuple[float, float]:
        """
        Web Mercator projection conversion (EPSG:3857 to EPSG:4326)
        Most accurate for satellite imagery
        """
        # Normalize to [0, 1]
        nx = x / self.width
        ny = y / self.height
        
        # Convert bounds to Web Mercator
        west_merc = self._lon_to_mercator_x(self.bounds['west'])
        east_merc = self._lon_to_mercator_x(self.bounds['east'])
        north_merc = self._lat_to_mercator_y(self.bounds['north'])
        south_merc = self._lat_to_mercator_y(self.bounds['south'])
        
        # Interpolate in Mercator space
        x_merc = west_merc + nx * (east_merc - west_merc)
        y_merc = north_merc - ny * (north_merc - south_merc)
        
        # Convert back to lat/lon
        lon = self._mercator_x_to_lon(x_merc)
        lat = self._mercator_y_to_lat(y_merc)
        
        return lat, lon
    
    @staticmethod
    def _lon_to_mercator_x(lon: float) -> float:
        """Convert longitude to Web Mercator X"""
        return lon * 20037508.34 / 180
    
    @staticmethod
    def _lat_to_mercator_y(lat: float) -> float:
        """Convert latitude to Web Mercator Y"""
        lat_rad = math.radians(lat)
        return math.log(math.tan((math.pi / 4) + (lat_rad / 2))) * 20037508.34 / math.pi
    
    @staticmethod
    def _mercator_x_to_lon(x: float) -> float:
        """Convert Web Mercator X to longitude"""
        return x * 180 / 20037508.34
    
    @staticmethod
    def _mercator_y_to_lat(y: float) -> float:
        """Convert Web Mercator Y to latitude"""
        lat_rad = 2 * math.atan(math.exp(y * math.pi / 20037508.34)) - math.pi / 2
        return math.degrees(lat_rad)
    
    def gps_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Convert GPS coordinates to pixel coordinates
        
        Args:
            lat, lon: GPS coordinates
            
        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        # Normalize to [0, 1]
        nx = (lon - self.bounds['west']) / (self.bounds['east'] - self.bounds['west'])
        ny = (self.bounds['north'] - lat) / (self.bounds['north'] - self.bounds['south'])
        
        # Convert to pixels
        x = int(np.clip(nx * self.width, 0, self.width - 1))
        y = int(np.clip(ny * self.height, 0, self.height - 1))
        
        return x, y
    
    def calculate_building_gps_footprint(self, building: Dict) -> Dict:
        """
        Calculate GPS coordinates for entire building footprint
        
        Args:
            building: Dict with 'bbox' and 'center_x', 'center_y'
            
        Returns:
            Dict with GPS coordinates for center and corners
        """
        bbox = building.get('bbox', {})
        x, y, w, h = bbox.get('x', 0), bbox.get('y', 0), \
                     bbox.get('width', 0), bbox.get('height', 0)
        
        # Calculate center
        center_lat, center_lon = self.pixel_to_gps(
            building.get('center_x', x + w/2),
            building.get('center_y', y + h/2),
            method='bilinear'
        )
        
        # Calculate corners
        corners_gps = []
        corner_pixels = [
            (x, y),           # Top-left
            (x + w, y),       # Top-right
            (x + w, y + h),   # Bottom-right
            (x, y + h)        # Bottom-left
        ]
        
        for px, py in corner_pixels:
            lat, lon = self.pixel_to_gps(px, py, method='bilinear')
            corners_gps.append({'lat': lat, 'lon': lon})
        
        # Calculate area in square meters
        area_m2 = building.get('area_pixels', 0) * self.pixel_area_m2
        
        return {
            'center': {'lat': center_lat, 'lon': center_lon},
            'corners': corners_gps,
            'area_m2': round(area_m2, 2),
            'perimeter_m': self._calculate_perimeter_meters(corners_gps)
        }
    
    def _calculate_perimeter_meters(self, corners: List[Dict]) -> float:
        """Calculate perimeter from GPS corners"""
        if len(corners) < 2:
            return 0
        
        perimeter = 0
        for i in range(len(corners)):
            p1 = corners[i]
            p2 = corners[(i + 1) % len(corners)]
            
            dist = self.haversine_distance(
                p1['lat'], p1['lon'],
                p2['lat'], p2['lon']
            )
            perimeter += dist
        
        return round(perimeter, 2)
    
    def calculate_accuracy_estimate(self, x: float, y: float) -> float:
        """
        Estimate GPS coordinate accuracy at a given pixel location
        
        Args:
            x, y: Pixel coordinates
            
        Returns:
            Estimated accuracy in meters (±)
        """
        # Accuracy degrades near edges and with distance from center
        center_x, center_y = self.width / 2, self.height / 2
        
        dx = abs(x - center_x) / center_x if center_x > 0 else 0
        dy = abs(y - center_y) / center_y if center_y > 0 else 0
        
        # Base accuracy (optimistic)
        base_accuracy = max(self.meters_per_pixel_x, self.meters_per_pixel_y)
        
        # Increase error near edges
        edge_factor = 1 + (dx + dy) / 2
        
        return round(base_accuracy * edge_factor, 2)
    
    def validate_gps_coordinates(self, lat: float, lon: float) -> bool:
        """
        Validate GPS coordinates are within image bounds
        
        Args:
            lat, lon: GPS coordinates to validate
            
        Returns:
            True if valid, False otherwise
        """
        return (self.bounds['south'] <= lat <= self.bounds['north'] and
                self.bounds['west'] <= lon <= self.bounds['east'])
    
    def export_geojson_point(self, lat: float, lon: float, properties: Dict = None) -> Dict:
        """
        Export GPS point as GeoJSON feature
        
        Args:
            lat, lon: GPS coordinates
            properties: Optional properties dict
            
        Returns:
            GeoJSON feature dict
        """
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]  # GeoJSON is [lon, lat]
            },
            "properties": properties or {}
        }
    
    def export_geojson_polygon(self, corners: List[Dict], properties: Dict = None) -> Dict:
        """
        Export building footprint as GeoJSON polygon
        
        Args:
            corners: List of dicts with 'lat' and 'lon'
            properties: Optional properties dict
            
        Returns:
            GeoJSON feature dict
        """
        # GeoJSON coordinates are [lon, lat]
        coordinates = [[corner['lon'], corner['lat']] for corner in corners]
        # Close the polygon
        coordinates.append(coordinates[0])
        
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates]
            },
            "properties": properties or {}
        }
    
    def get_statistics(self) -> Dict:
        """Get calculator statistics"""
        return {
            'image_size': {'width': self.width, 'height': self.height},
            'bounds': self.bounds,
            'center': {'lat': self.center_lat, 'lon': self.center_lon},
            'resolution': {
                'meters_per_pixel_x': round(self.meters_per_pixel_x, 3),
                'meters_per_pixel_y': round(self.meters_per_pixel_y, 3),
                'pixel_area_m2': round(self.pixel_area_m2, 3)
            },
            'coverage_area_km2': round(
                (self.width * self.meters_per_pixel_x * 
                 self.height * self.meters_per_pixel_y) / 1_000_000, 3
            )
        }


# Integration function for your existing code
def enhance_building_with_accurate_gps(building: Dict, gps_calc: GPSCoordinateCalculator) -> Dict:
    """
    Enhance building metadata with accurate GPS coordinates
    
    Args:
        building: Building dict with pixel coordinates
        gps_calc: GPSCoordinateCalculator instance
        
    Returns:
        Enhanced building dict with GPS data
    """
    # Calculate GPS footprint
    gps_data = gps_calc.calculate_building_gps_footprint(building)
    
    # Update building dict
    building.update({
        'gps_latitude': gps_data['center']['lat'],
        'gps_longitude': gps_data['center']['lon'],
        'gps_corners': gps_data['corners'],
        'area_m2': gps_data['area_m2'],
        'perimeter_m': gps_data['perimeter_m'],
        'gps_accuracy_estimate_m': gps_calc.calculate_accuracy_estimate(
            building.get('center_x', 0),
            building.get('center_y', 0)
        )
    })
    
    # Create GeoJSON representations
    building['geojson_point'] = gps_calc.export_geojson_point(
        building['gps_latitude'],
        building['gps_longitude'],
        properties={'building_id': building.get('id')}
    )
    
    building['geojson_polygon'] = gps_calc.export_geojson_polygon(
        gps_data['corners'],
        properties={
            'building_id': building.get('id'),
            'area_m2': gps_data['area_m2'],
            'status': building.get('legal_status', 'UNKNOWN')
        }
    )
    
    return building


# Example usage
if __name__ == "__main__":
    # Example: Mysuru region
    bounds = {
        'north': 12.3200,
        'south': 12.2900,
        'east': 76.6600,
        'west': 76.6300
    }
    
    image_size = (2048, 2048)  # Example image size
    
    # Create calculator
    gps_calc = GPSCoordinateCalculator(bounds, image_size)
    
    # Print statistics
    stats = gps_calc.get_statistics()
    print("\n=== GPS Calculator Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test conversion
    print("\n=== Test Conversions ===")
    
    # Center of image
    lat, lon = gps_calc.pixel_to_gps(1024, 1024, method='bilinear')
    print(f"Center (1024, 1024) -> ({lat:.6f}, {lon:.6f})")
    
    # Top-left corner
    lat, lon = gps_calc.pixel_to_gps(0, 0, method='bilinear')
    print(f"Top-left (0, 0) -> ({lat:.6f}, {lon:.6f})")
    
    # Example building
    building = {
        'id': 'TEST_001',
        'center_x': 1024,
        'center_y': 1024,
        'area_pixels': 500,
        'bbox': {'x': 1000, 'y': 1000, 'width': 48, 'height': 48}
    }
    
    enhanced = enhance_building_with_accurate_gps(building, gps_calc)
    print(f"\n=== Enhanced Building ===")
    print(f"GPS Center: ({enhanced['gps_latitude']:.6f}, {enhanced['gps_longitude']:.6f})")
    print(f"Area: {enhanced['area_m2']:.2f} m²")
    print(f"Perimeter: {enhanced['perimeter_m']:.2f} m")
    print(f"Accuracy: ±{enhanced['gps_accuracy_estimate_m']:.2f} m")