import jwt
import logging
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify
from typing import Dict, Optional, Callable

logger = logging.getLogger(__name__)

class AuthManager:
    """Handles authentication and authorization using JWT tokens"""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256', 
                 token_expiry_hours: int = 24):
        """
        Initialize AuthManager
        
        Args:
            secret_key: Secret key for JWT encoding/decoding
            algorithm: JWT algorithm (default: HS256)
            token_expiry_hours: Token expiration time in hours (default: 24)
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry_hours = token_expiry_hours
        logger.info("AuthManager initialized")
    
    def generate_token(self, user_id: str, username: str, 
                      role: str = 'user', **extra_claims) -> str:
        """
        Generate a JWT token for a user
        
        Args:
            user_id: Unique user identifier
            username: Username
            role: User role (e.g., 'admin', 'user')
            **extra_claims: Additional claims to include in token
            
        Returns:
            JWT token string
        """
        payload = {
            'user_id': user_id,
            'username': username,
            'role': role,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours)
        }
        
        # Add any extra claims
        payload.update(extra_claims)
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Token generated for user: {username}")
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """
        Verify and decode a JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def token_required(self, f: Callable) -> Callable:
        """
        Decorator to protect routes with JWT authentication
        
        Usage:
            @app.route('/protected')
            @auth_manager.token_required
            def protected_route(current_user):
                return jsonify({'message': 'Access granted'})
        """
        @wraps(f)
        def decorated(*args, **kwargs):
            token = None
            
            # Check for token in Authorization header
            if 'Authorization' in request.headers:
                auth_header = request.headers['Authorization']
                try:
                    # Expected format: "Bearer <token>"
                    token = auth_header.split(" ")[1]
                except IndexError:
                    return jsonify({'error': 'Invalid authorization header format'}), 401
            
            if not token:
                return jsonify({'error': 'Authentication token is missing'}), 401
            
            # Verify token
            current_user = self.verify_token(token)
            
            if not current_user:
                return jsonify({'error': 'Invalid or expired token'}), 401
            
            # Pass current_user to the route function
            return f(current_user, *args, **kwargs)
        
        return decorated
    
    def role_required(self, required_role: str) -> Callable:
        """
        Decorator to restrict access based on user role
        
        Usage:
            @app.route('/admin')
            @auth_manager.role_required('admin')
            def admin_route(current_user):
                return jsonify({'message': 'Admin access granted'})
        """
        def decorator(f: Callable) -> Callable:
            @wraps(f)
            def decorated(*args, **kwargs):
                token = None
                
                # Check for token in Authorization header
                if 'Authorization' in request.headers:
                    auth_header = request.headers['Authorization']
                    try:
                        token = auth_header.split(" ")[1]
                    except IndexError:
                        return jsonify({'error': 'Invalid authorization header format'}), 401
                
                if not token:
                    return jsonify({'error': 'Authentication token is missing'}), 401
                
                # Verify token
                current_user = self.verify_token(token)
                
                if not current_user:
                    return jsonify({'error': 'Invalid or expired token'}), 401
                
                # Check role
                if current_user.get('role') != required_role:
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                return f(current_user, *args, **kwargs)
            
            return decorated
        return decorator
    
    def refresh_token(self, token: str) -> Optional[str]:
        """
        Refresh an existing token
        
        Args:
            token: Existing JWT token
            
        Returns:
            New JWT token if valid, None otherwise
        """
        payload = self.verify_token(token)
        
        if not payload:
            return None
        
        # Generate new token with same claims
        new_token = self.generate_token(
            user_id=payload['user_id'],
            username=payload['username'],
            role=payload.get('role', 'user')
        )
        
        logger.info(f"Token refreshed for user: {payload['username']}")
        return new_token
    
    def get_user_from_request(self) -> Optional[Dict]:
        """
        Extract user information from current request
        
        Returns:
            User payload if token is valid, None otherwise
        """
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return None
        
        if not token:
            return None
        
        return self.verify_token(token)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password (basic implementation)
        For production, use bcrypt or passlib
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash
        
        Args:
            password: Plain text password
            hashed_password: Hashed password to compare against
            
        Returns:
            True if password matches, False otherwise
        """
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest() == hashed_password


# Example usage for user management (optional)
class UserManager:
    """Simple user management system (can be extended)"""
    
    def __init__(self, db_manager):
        """
        Initialize UserManager
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
        self._create_users_table()
    
    def _create_users_table(self):
        """Create users table if it doesn't exist"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
            conn.commit()
    
    def create_user(self, username: str, password: str, email: str = None, 
                   role: str = 'user') -> Optional[str]:
        """Create a new user"""
        import uuid
        
        user_id = str(uuid.uuid4())
        password_hash = AuthManager.hash_password(password)
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users (user_id, username, password_hash, email, role)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, username, password_hash, email, role))
                conn.commit()
                logger.info(f"User created: {username}")
                return user_id
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate a user"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_id, username, password_hash, role 
                    FROM users WHERE username = ?
                ''', (username,))
                
                user = cursor.fetchone()
                
                if not user:
                    logger.warning(f"User not found: {username}")
                    return None
                
                if AuthManager.verify_password(password, user['password_hash']):
                    # Update last login
                    cursor.execute('''
                        UPDATE users SET last_login = CURRENT_TIMESTAMP 
                        WHERE username = ?
                    ''', (username,))
                    conn.commit()
                    
                    logger.info(f"User authenticated: {username}")
                    return {
                        'user_id': user['user_id'],
                        'username': user['username'],
                        'role': user['role']
                    }
                else:
                    logger.warning(f"Invalid password for user: {username}")
                    return None
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user information"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_id, username, email, role, created_at, last_login
                    FROM users WHERE username = ?
                ''', (username,))
                
                user = cursor.fetchone()
                return dict(user) if user else None
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None