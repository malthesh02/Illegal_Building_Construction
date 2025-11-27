
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import DATABASE_PATH, setup_logging
from database_manager import DatabaseManager
from auth_manager import UserManager
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def initialize_database():
    """Initialize the database with all necessary tables"""
    
    print("=" * 70)
    print("DATABASE INITIALIZATION")
    print("=" * 70)
    
    # Ensure data directory exists
    db_dir = os.path.dirname(DATABASE_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        print(f"✓ Created directory: {db_dir}")
    
    # Initialize DatabaseManager (creates tables)
    print(f"\n📦 Initializing database: {DATABASE_PATH}")
    db_manager = DatabaseManager(DATABASE_PATH)
    print("✓ Database tables created")
    
    # Initialize UserManager (creates users table)
    print("\n👤 Initializing user management...")
    user_manager = UserManager(db_manager)
    print("✓ Users table verified")
    
    # Verify all tables
    print("\n🔍 Verifying database structure...")
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['sessions', 'results', 'buildings', 'cadastral_permits', 'users']
        
        print("\nDatabase Tables:")
        for table in tables:
            status = "✓" if table in expected_tables else "⚠"
            print(f"  {status} {table}")
        
        # Check users table structure
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        print("\nUsers Table Structure:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    
    # Create admin user (optional)
    print("\n" + "=" * 70)
    create_admin = input("Do you want to create an admin user? (y/n): ").lower()
    
    if create_admin == 'y':
        username = input("Enter admin username: ").strip()
        password = input("Enter admin password (min 6 chars): ").strip()
        email = input("Enter admin email: ").strip()
        
        if len(password) < 6:
            print("❌ Password must be at least 6 characters!")
        elif len(username) < 3:
            print("❌ Username must be at least 3 characters!")
        else:
            user_id = user_manager.create_user(
                username=username,
                password=password,
                email=email,
                role='admin'
            )
            
            if user_id:
                print(f"✅ Admin user created: {username}")
            else:
                print("❌ Failed to create admin user (might already exist)")
    
    print("\n" + "=" * 70)
    print("✅ DATABASE INITIALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nDatabase location: {DATABASE_PATH}")
    print(f"Database size: {os.path.getsize(DATABASE_PATH)} bytes")
    
    # Show statistics
    stats = db_manager.get_overall_statistics()
    print("\nCurrent Statistics:")
    print(f"  - Users: {get_user_count(db_manager)}")
    print(f"  - Sessions: {stats.get('total_sessions', 0)}")
    print(f"  - Buildings: {stats.get('total_buildings', 0)}")
    print(f"  - Permits: {stats.get('total_permits', 0)}")
    
    print("\n🚀 You can now start your Flask application!")
    print("   Run: python app.py")

def get_user_count(db_manager):
    """Get total user count"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM users")
            return cursor.fetchone()['count']
    except:
        return 0

if __name__ == "__main__":
    initialize_database()