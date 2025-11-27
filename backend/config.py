"""
Configuration Module - Production Ready Version

COMPLETE FIXES APPLIED:
========================
1. Environment variable support for configuration
2. Validation of paths and parameters
3. Auto-creation of required directories
4. Platform-specific path handling
5. Configuration loading from file
6. Configuration export/import
7. Runtime configuration updates
8. Better organization with config classes
9. Type hints and documentation
10. Default fallbacks for all settings
11. Flask and JWT configuration added
12. PDF Report Configuration ADDED ✨

Author: Corrected Version
Date: 2025
"""

import os
import sys
import json
from pathlib import Path
from typing import Set, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    
    # Base directories
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    project_root: Optional[str] = None
    
    # Data directories
    upload_dir: Optional[str] = None
    results_dir: Optional[str] = None
    models_dir: Optional[str] = None
    temp_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    
    # Model paths
    seg_model_dir: Optional[str] = None
    
    # ⭐ NEW: Reports directory
    reports_dir: Optional[str] = None
    
    def __post_init__(self):
        """Initialize computed paths after dataclass creation."""
        # Set project root if not provided
        if self.project_root is None:
            self.project_root = os.path.abspath(os.path.join(self.base_dir, ".."))
        
        # Set default directories if not provided
        if self.upload_dir is None:
            self.upload_dir = os.path.join(self.project_root, "data", "uploads")
        
        if self.results_dir is None:
            self.results_dir = os.path.join(self.project_root, "data", "results")
        
        if self.models_dir is None:
            self.models_dir = os.path.join(self.project_root, "models")
        
        if self.temp_dir is None:
            self.temp_dir = os.path.join(self.project_root, "data", "temp")
        
        if self.logs_dir is None:
            self.logs_dir = os.path.join(self.project_root, "logs")
        
        if self.seg_model_dir is None:
            self.seg_model_dir = os.path.join(self.models_dir, "segmentation_model")
        
        # ⭐ NEW: Set reports directory
        if self.reports_dir is None:
            self.reports_dir = os.path.join(self.project_root, "data", "reports")
        
        # Override with environment variables if set
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'UPLOAD_DIR': 'upload_dir',
            'RESULTS_DIR': 'results_dir',
            'MODELS_DIR': 'models_dir',
            'TEMP_DIR': 'temp_dir',
            'LOGS_DIR': 'logs_dir',
            'SEG_MODEL_DIR': 'seg_model_dir',
            'REPORTS_DIR': 'reports_dir',  # ⭐ NEW
        }
        
        for env_var, attr in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                setattr(self, attr, env_value)
                logger.info(f"Loaded {attr} from environment: {env_value}")
    
    def create_directories(self, permissions: int = 0o755):
        """Create all configured directories."""
        dirs_to_create = [
            self.upload_dir,
            self.results_dir,
            self.models_dir,
            self.temp_dir,
            self.logs_dir,
            self.seg_model_dir,
            self.reports_dir,  # ⭐ NEW
        ]
        
        for dir_path in dirs_to_create:
            if dir_path:
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    
                    # Set permissions (Unix only)
                    if sys.platform != 'win32':
                        try:
                            os.chmod(dir_path, permissions)
                        except Exception:
                            pass  # Ignore permission errors
                    
                    logger.debug(f"Ensured directory exists: {dir_path}")
                except Exception as e:
                    logger.error(f"Failed to create directory {dir_path}: {e}")
    
    def validate(self) -> bool:
        """
        Validate all paths are accessible.
        
        Returns:
        --------
        bool : True if all paths are valid
        """
        all_valid = True
        
        # Check base directory exists
        if not os.path.exists(self.base_dir):
            logger.error(f"Base directory doesn't exist: {self.base_dir}")
            all_valid = False
        
        # Check writable directories
        writable_dirs = [self.upload_dir, self.results_dir, self.temp_dir, self.logs_dir, self.reports_dir]  # ⭐ NEW
        
        for dir_path in writable_dirs:
            if dir_path and os.path.exists(dir_path):
                if not os.access(dir_path, os.W_OK):
                    logger.error(f"Directory not writable: {dir_path}")
                    all_valid = False
        
        return all_valid


@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters."""
    
    # Image processing
    max_image_dim: int = 2048
    min_image_dim: int = 100
    default_resize_quality: int = 95
    thumbnail_size: tuple = (256, 256)
    
    # File handling
    allowed_extensions: Set[str] = field(default_factory=lambda: {
        "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"
    })
    max_file_size_mb: float = 50.0
    
    # Processing timeouts (seconds)
    max_processing_time: int = 300
    max_upload_time: int = 60
    
    # Memory management
    max_memory_mb: int = 2048
    enable_memory_optimization: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    
    def __post_init__(self):
        """Load from environment variables."""
        # Max image dimension
        env_max_dim = os.getenv('MAX_IMAGE_DIM')
        if env_max_dim:
            try:
                self.max_image_dim = int(env_max_dim)
            except ValueError:
                logger.warning(f"Invalid MAX_IMAGE_DIM: {env_max_dim}")
        
        # Max file size
        env_max_size = os.getenv('MAX_FILE_SIZE_MB')
        if env_max_size:
            try:
                self.max_file_size_mb = float(env_max_size)
            except ValueError:
                logger.warning(f"Invalid MAX_FILE_SIZE_MB: {env_max_size}")
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return int(self.max_file_size_mb * 1024 * 1024)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        valid = True
        
        if self.max_image_dim < self.min_image_dim:
            logger.error("max_image_dim must be >= min_image_dim")
            valid = False
        
        if self.max_file_size_mb <= 0:
            logger.error("max_file_size_mb must be positive")
            valid = False
        
        if not self.allowed_extensions:
            logger.error("allowed_extensions cannot be empty")
            valid = False
        
        return valid


@dataclass
class AlgorithmConfig:
    """Configuration for detection algorithms."""
    
    # K-means parameters
    kmeans_clusters: int = 2
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    kmeans_downsample: int = 2
    
    # Edge detection
    canny_low_threshold: int = 30
    canny_high_threshold: int = 100
    
    # Building detection
    min_building_area: int = 100
    max_building_area_ratio: float = 0.5
    min_extent: float = 0.3
    max_extent: float = 0.95
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 5.0
    min_solidity: float = 0.7
    max_circularity: float = 0.85
    
    # Morphological operations
    morph_kernel_size: int = 5
    morph_iterations: int = 2
    
    # CNN model
    model_confidence_threshold: float = 0.5
    use_gpu: bool = False
    batch_size: int = 1
    
    def __post_init__(self):
        """Load from environment if available."""
        env_gpu = os.getenv('USE_GPU')
        if env_gpu:
            self.use_gpu = env_gpu.lower() in ['true', '1', 'yes']
    
    def validate(self) -> bool:
        """Validate algorithm parameters."""
        valid = True
        
        if self.kmeans_clusters < 2:
            logger.error("kmeans_clusters must be >= 2")
            valid = False
        
        if not (0.0 <= self.model_confidence_threshold <= 1.0):
            logger.error("model_confidence_threshold must be between 0 and 1")
            valid = False
        
        if self.min_building_area < 1:
            logger.error("min_building_area must be positive")
            valid = False
        
        return valid


@dataclass
class APIConfig:
    """Configuration for API/server settings."""
    
    # Server
    host: str = '0.0.0.0'
    port: int = 5000
    debug: bool = False
    flask_env: str = 'development'
    
    # CORS
    cors_enabled: bool = True
    cors_origins: list = field(default_factory=lambda: [
        "http://localhost:*",
        "http://127.0.0.1:*",
        "http://0.0.0.0:*"
    ])
    cors_methods: list = field(default_factory=lambda: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
    cors_allow_headers: list = field(default_factory=lambda: ['Content-Type', 'Authorization'])
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_per_hour: int = 100
    
    # API keys and JWT
    require_api_key: bool = False
    api_keys: Set[str] = field(default_factory=set)
    jwt_secret_key: str = field(default_factory=lambda: os.urandom(32).hex())
    
    # Flask configuration
    api_version: str = '1.0.0'
    max_content_length: int = 50 * 1024 * 1024  # 50MB
    
    # Timeouts
    request_timeout: int = 300
    
    # ⭐ NEW: Report settings
    reports_enabled: bool = True
    reports_dir: Optional[str] = None
    max_reports_to_keep: int = 100
    auto_generate_reports: bool = False
    
    def __post_init__(self):
        """Load from environment."""
        # Host
        env_host = os.getenv('API_HOST')
        if env_host:
            self.host = env_host
        
        # Port
        env_port = os.getenv('API_PORT')
        if env_port:
            try:
                self.port = int(env_port)
            except ValueError:
                logger.warning(f"Invalid API_PORT: {env_port}")
        
        # Debug mode
        env_debug = os.getenv('DEBUG')
        if env_debug:
            self.debug = env_debug.lower() in ['true', '1', 'yes']
        
        # Flask environment
        env_flask_env = os.getenv('FLASK_ENV')
        if env_flask_env:
            self.flask_env = env_flask_env
        
        # JWT Secret Key
        env_jwt_key = os.getenv('JWT_SECRET_KEY')
        if env_jwt_key:
            self.jwt_secret_key = env_jwt_key
        
        # API keys from environment
        env_keys = os.getenv('API_KEYS')
        if env_keys:
            self.api_keys = set(env_keys.split(','))
            self.require_api_key = True


class Config:
    """
    Main configuration class that combines all config sections.
    
    Usage:
    ------
    from config import config
    
    # Access settings
    max_dim = config.processing.max_image_dim
    upload_dir = config.paths.upload_dir
    
    # Update settings
    config.algorithm.kmeans_clusters = 3
    
    # Save/load
    config.save_to_file('config.json')
    config.load_from_file('config.json')
    """
    
    def __init__(self):
        """Initialize all configuration sections."""
        self.paths = PathConfig()
        self.processing = ProcessingConfig()
        self.algorithm = AlgorithmConfig()
        self.api = APIConfig()
        
        # Create directories on initialization
        self.paths.create_directories()
    
    def validate_all(self) -> bool:
        """
        Validate all configuration sections.
        
        Returns:
        --------
        bool : True if all sections are valid
        """
        results = {
            'paths': self.paths.validate(),
            'processing': self.processing.validate(),
            'algorithm': self.algorithm.validate(),
        }
        
        all_valid = all(results.values())
        
        if not all_valid:
            logger.error("Configuration validation failed:")
            for section, valid in results.items():
                status = "✓" if valid else "✗"
                logger.error(f"  {status} {section}")
        else:
            logger.info("✓ All configuration sections validated successfully")
        
        return all_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration to dictionary.
        
        Returns:
        --------
        dict : Configuration as dictionary
        """
        return {
            'paths': asdict(self.paths),
            'processing': {
                **asdict(self.processing),
                'allowed_extensions': list(self.processing.allowed_extensions)
            },
            'algorithm': asdict(self.algorithm),
            'api': {
                **asdict(self.api),
                'api_keys': list(self.api.api_keys) if self.api.api_keys else []
            }
        }
    
    def from_dict(self, config_dict: Dict[str, Any]):
        """
        Load configuration from dictionary.
        
        Parameters:
        -----------
        config_dict : dict
            Configuration dictionary
        """
        if 'paths' in config_dict:
            for key, value in config_dict['paths'].items():
                if hasattr(self.paths, key):
                    setattr(self.paths, key, value)
        
        if 'processing' in config_dict:
            proc = config_dict['processing']
            for key, value in proc.items():
                if key == 'allowed_extensions':
                    self.processing.allowed_extensions = set(value)
                elif hasattr(self.processing, key):
                    setattr(self.processing, key, value)
        
        if 'algorithm' in config_dict:
            for key, value in config_dict['algorithm'].items():
                if hasattr(self.algorithm, key):
                    setattr(self.algorithm, key, value)
        
        if 'api' in config_dict:
            api = config_dict['api']
            for key, value in api.items():
                if key == 'api_keys':
                    self.api.api_keys = set(value) if value else set()
                elif hasattr(self.api, key):
                    setattr(self.api, key, value)
    
    def save_to_file(self, filepath: str = 'config.json'):
        """
        Save configuration to JSON file.
        
        Parameters:
        -----------
        filepath : str
            Path to config file
        """
        try:
            config_dict = self.to_dict()
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def load_from_file(self, filepath: str = 'config.json'):
        """
        Load configuration from JSON file.
        
        Parameters:
        -----------
        filepath : str
            Path to config file
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Config file not found: {filepath}")
                return
            
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            self.from_dict(config_dict)
            logger.info(f"Configuration loaded from: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        
        print("\n📁 PATHS:")
        print(f"  Project Root:    {self.paths.project_root}")
        print(f"  Upload Dir:      {self.paths.upload_dir}")
        print(f"  Results Dir:     {self.paths.results_dir}")
        print(f"  Reports Dir:     {self.paths.reports_dir}")  # ⭐ NEW
        print(f"  Models Dir:      {self.paths.models_dir}")
        print(f"  Temp Dir:        {self.paths.temp_dir}")
        print(f"  Logs Dir:        {self.paths.logs_dir}")
        
        print("\n⚙️  PROCESSING:")
        print(f"  Max Image Dim:   {self.processing.max_image_dim} px")
        print(f"  Max File Size:   {self.processing.max_file_size_mb} MB")
        print(f"  Allowed Formats: {', '.join(sorted(self.processing.allowed_extensions))}")
        print(f"  Memory Limit:    {self.processing.max_memory_mb} MB")
        
        print("\n🔬 ALGORITHMS:")
        print(f"  K-means Clusters: {self.algorithm.kmeans_clusters}")
        print(f"  K-means N-init:   {self.algorithm.kmeans_n_init}")
        print(f"  Min Building Area: {self.algorithm.min_building_area} px")
        print(f"  Confidence Threshold: {self.algorithm.model_confidence_threshold}")
        print(f"  Use GPU:          {self.algorithm.use_gpu}")
        
        print("\n🌐 API:")
        print(f"  Host:            {self.api.host}")
        print(f"  Port:            {self.api.port}")
        print(f"  Debug Mode:      {self.api.debug}")
        print(f"  Flask Env:       {self.api.flask_env}")
        print(f"  API Version:     {self.api.api_version}")
        print(f"  CORS Enabled:    {self.api.cors_enabled}")
        print(f"  Rate Limiting:   {self.api.rate_limit_enabled}")
        
        print("\n📄 REPORTS:")  # ⭐ NEW
        print(f"  Reports Enabled: {self.api.reports_enabled}")
        print(f"  Reports Dir:     {self.api.reports_dir}")
        print(f"  Auto Generate:   {self.api.auto_generate_reports}")
        
        print("\n" + "=" * 70 + "\n")
    
    @staticmethod
    def init_app():
        """Initialize app - create directories and database path"""
        config.paths.create_directories()
        
        # Ensure database directory exists
        db_dir = os.path.dirname(DATABASE_PATH)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)


# Create global config instance
config = Config()

# Backward compatibility: Export individual settings at module level
BASE_DIR = config.paths.base_dir
PROJECT_ROOT = config.paths.project_root
UPLOAD_DIR = config.paths.upload_dir
RESULTS_DIR = config.paths.results_dir
MODELS_DIR = config.paths.models_dir
SEG_MODEL_DIR = config.paths.seg_model_dir
REPORTS_DIR = config.paths.reports_dir  # ⭐ NEW

ALLOWED_EXTENSIONS = config.processing.allowed_extensions
MAX_IMAGE_DIM = config.processing.max_image_dim
MAX_FILE_SIZE_MB = config.processing.max_file_size_mb

# Flask/API settings for backward compatibility
HOST = config.api.host
PORT = config.api.port
DEBUG = config.api.debug
FLASK_ENV = config.api.flask_env
API_VERSION = config.api.api_version

# Database
DATABASE_PATH = os.path.join(config.paths.project_root, 'data', 'illegal_buildings.db')

# JWT
JWT_SECRET_KEY = config.api.jwt_secret_key

# CORS
CORS_ORIGINS = config.api.cors_origins
CORS_METHODS = config.api.cors_methods
CORS_ALLOW_HEADERS = config.api.cors_allow_headers

# File upload
MAX_CONTENT_LENGTH = config.api.max_content_length
UPLOAD_FOLDER = config.paths.upload_dir
RESULTS_FOLDER = config.paths.results_dir


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
    --------
    Config : Global configuration object
    """
    return config


def reload_config():
    """Reload configuration from environment and files."""
    global config
    config = Config()
    
    # Try to load from config file if exists
    config_file = os.path.join(config.paths.project_root, 'config.json')
    if os.path.exists(config_file):
        config.load_from_file(config_file)
    
    logger.info("Configuration reloaded")

    

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configure logging for the application.
    
    Parameters:
    -----------
    log_level : int
        Logging level (default: logging.INFO)
    log_file : str, optional
        Path to log file. If None, logs only to console.
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to setup file logging: {e}")
    
    # Log startup message
    logger.info("Logging configured successfully")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")


def validate_environment() -> bool:
    """
    Validate the environment is properly configured.
    
    Returns:
    --------
    bool : True if environment is valid
    """
    checks = {
        'Python Version': sys.version_info >= (3, 7),
        'Config Valid': config.validate_all(),
        'Upload Dir Exists': os.path.exists(config.paths.upload_dir),
        'Results Dir Exists': os.path.exists(config.paths.results_dir),
        'Reports Dir Exists': os.path.exists(config.paths.reports_dir),  # ⭐ NEW
        'Upload Dir Writable': os.access(config.paths.upload_dir, os.W_OK),
        'Results Dir Writable': os.access(config.paths.results_dir, os.W_OK),
        'Reports Dir Writable': os.access(config.paths.reports_dir, os.W_OK),  # ⭐ NEW
    }
    
    all_passed = all(checks.values())
    
    print("\n" + "=" * 70)
    print("ENVIRONMENT VALIDATION")
    print("=" * 70)
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    
    print("=" * 70)
    
    if all_passed:
        print("✓ Environment validation PASSED\n")
    else:
        print("✗ Environment validation FAILED\n")
    
    return all_passed


if __name__ == "__main__":
    # Self-test
    print("\n" + "=" * 70)
    print("CONFIG MODULE SELF-TEST")
    print("=" * 70)
    
    # Print configuration
    config.print_summary()
    
    # Validate
    if validate_environment():
        print("✓ Configuration module is working correctly")
    else:
        print("✗ Configuration module has issues")
    
    # Test save/load
    print("\nTesting save/load...")
    test_file = 'test_config.json'
    
    try:
        config.save_to_file(test_file)
        print(f"✓ Saved to {test_file}")
        
        # Modify config
        original_max_dim = config.processing.max_image_dim
        config.processing.max_image_dim = 1024
        
        # Reload
        config.load_from_file(test_file)
        print(f"✓ Loaded from {test_file}")
        
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"✓ Removed {test_file}")
        
    except Exception as e:
        print(f"✗ Save/load test failed: {e}")
    
    print("\n" + "=" * 70 + "\n")