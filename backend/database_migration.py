
import sqlite3
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def find_database():
    """Find the database file automatically"""
    
    # Check common locations
    possible_paths = [
        'data/illegal_buildings.db',
        'illegal_buildings.db',
        '../data/illegal_buildings.db'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"✅ Found database at: {path}")
            return path
    
    # Ask user
    logger.warning("⚠️ Could not find database automatically")
    db_path = input("Enter the path to your database file: ").strip()
    
    if os.path.exists(db_path):
        return db_path
    else:
        logger.error(f"❌ Database not found at: {db_path}")
        sys.exit(1)


def backup_database(db_path):
    """Create a backup before modifying"""
    import shutil
    from datetime import datetime
    
    backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        shutil.copy2(db_path, backup_path)
        logger.info(f"✅ Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        return None


def column_exists(cursor, table_name, column_name):
    """Check if a column already exists in a table"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def migrate_database(db_path):
    """
    Add new columns for two-temporal change detection
    
    New columns added:
    - buildings.change_percentage: Change % for this building
    - buildings.construction_status: 'EXISTING' or 'UNDER_CONSTRUCTION'
    - buildings.requires_field_visit: Flag for field inspection
    - sessions.change_threshold: Threshold used for detection
    """
    
    logger.info("\n" + "="*70)
    logger.info("STARTING DATABASE MIGRATION")
    logger.info("="*70 + "\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    changes_made = 0
    
    try:
        # ===== Modify BUILDINGS table =====
        logger.info("📋 Updating BUILDINGS table...")
        
        # 1. Add change_percentage column
        if not column_exists(cursor, 'buildings', 'change_percentage'):
            logger.info("  Adding 'change_percentage' column...")
            cursor.execute("""
                ALTER TABLE buildings 
                ADD COLUMN change_percentage REAL DEFAULT 0.0
            """)
            logger.info("  ✅ Added 'change_percentage'")
            changes_made += 1
        else:
            logger.info("  ⏭️ 'change_percentage' already exists")
        
        # 2. Add construction_status column
        if not column_exists(cursor, 'buildings', 'construction_status'):
            logger.info("  Adding 'construction_status' column...")
            cursor.execute("""
                ALTER TABLE buildings 
                ADD COLUMN construction_status TEXT DEFAULT 'UNKNOWN'
            """)
            logger.info("  ✅ Added 'construction_status'")
            changes_made += 1
        else:
            logger.info("  ⏭️ 'construction_status' already exists")
        
        # 3. Add requires_field_visit column
        if not column_exists(cursor, 'buildings', 'requires_field_visit'):
            logger.info("  Adding 'requires_field_visit' column...")
            cursor.execute("""
                ALTER TABLE buildings 
                ADD COLUMN requires_field_visit BOOLEAN DEFAULT 0
            """)
            logger.info("  ✅ Added 'requires_field_visit'")
            changes_made += 1
        else:
            logger.info("  ⏭️ 'requires_field_visit' already exists")
        
        # ===== Modify SESSIONS table =====
        logger.info("\n📋 Updating SESSIONS table...")
        
        # 4. Add change_threshold column
        if not column_exists(cursor, 'sessions', 'change_threshold'):
            logger.info("  Adding 'change_threshold' column...")
            cursor.execute("""
                ALTER TABLE sessions 
                ADD COLUMN change_threshold REAL DEFAULT 30.0
            """)
            logger.info("  ✅ Added 'change_threshold'")
            changes_made += 1
        else:
            logger.info("  ⏭️ 'change_threshold' already exists")
        
        # ===== Create new table for change metadata =====
        logger.info("\n📋 Creating CHANGE_DETECTION_METADATA table...")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS change_detection_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                total_change_pixels INTEGER,
                change_percentage REAL,
                num_change_regions INTEGER,
                largest_change_area INTEGER,
                change_threshold_used REAL DEFAULT 30.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        logger.info("  ✅ Created 'change_detection_metadata' table")
        changes_made += 1
        
        # Commit changes
        conn.commit()
        
        logger.info("\n" + "="*70)
        logger.info(f"✅ MIGRATION COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 Total changes made: {changes_made}")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Migration failed: {e}")
        conn.rollback()
        raise
    
    finally:
        conn.close()
    
    return changes_made


def verify_migration(db_path):
    """Verify all changes were applied correctly"""
    
    logger.info("\n" + "="*70)
    logger.info("VERIFYING DATABASE SCHEMA")
    logger.info("="*70 + "\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    all_good = True
    
    # Check buildings table
    logger.info("📋 Checking BUILDINGS table:")
    required_cols = ['change_percentage', 'construction_status', 'requires_field_visit']
    
    cursor.execute("PRAGMA table_info(buildings)")
    buildings_columns = [row[1] for row in cursor.fetchall()]
    
    for col in required_cols:
        if col in buildings_columns:
            logger.info(f"  ✅ {col}")
        else:
            logger.error(f"  ❌ {col} - MISSING!")
            all_good = False
    
    # Check sessions table
    logger.info("\n📋 Checking SESSIONS table:")
    cursor.execute("PRAGMA table_info(sessions)")
    sessions_columns = [row[1] for row in cursor.fetchall()]
    
    if 'change_threshold' in sessions_columns:
        logger.info("  ✅ change_threshold")
    else:
        logger.error("  ❌ change_threshold - MISSING!")
        all_good = False
    
    # Check change_detection_metadata table
    logger.info("\n📋 Checking CHANGE_DETECTION_METADATA table:")
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='change_detection_metadata'
    """)
    
    if cursor.fetchone():
        logger.info("  ✅ Table exists")
        
        # Check its columns
        cursor.execute("PRAGMA table_info(change_detection_metadata)")
        columns = [row[1] for row in cursor.fetchall()]
        expected = ['id', 'session_id', 'total_change_pixels', 'change_percentage', 
                   'num_change_regions', 'largest_change_area', 'change_threshold_used', 'created_at']
        
        for col in expected:
            if col in columns:
                logger.info(f"    ✅ {col}")
            else:
                logger.error(f"    ❌ {col} - MISSING!")
                all_good = False
    else:
        logger.error("  ❌ Table does not exist!")
        all_good = False
    
    conn.close()
    
    logger.info("\n" + "="*70)
    if all_good:
        logger.info("✅ ALL CHECKS PASSED - Database is ready!")
    else:
        logger.error("❌ SOME CHECKS FAILED - Please review errors above")
    logger.info("="*70 + "\n")
    
    return all_good


def show_current_schema(db_path):
    """Display current database schema"""
    
    logger.info("\n" + "="*70)
    logger.info("CURRENT DATABASE SCHEMA")
    logger.info("="*70 + "\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    for table in tables:
        logger.info(f"📋 Table: {table}")
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        for col in columns:
            col_id, name, type_, notnull, default, pk = col
            logger.info(f"  - {name:30s} {type_:15s} {'PRIMARY KEY' if pk else ''}")
        
        logger.info("")
    
    conn.close()


def main():
    """Main migration workflow"""
    
    print("\n" + "="*70)
    print("🔧 DATABASE MIGRATION TOOL")
    print("   Two-Temporal Change Detection Support")
    print("="*70 + "\n")
    
    # Find database
    db_path = find_database()
    
    # Show current schema
    show_schema = input("\n📊 Show current database schema? (y/n): ").strip().lower()
    if show_schema == 'y':
        show_current_schema(db_path)
    
    # Confirm migration
    print("\n⚠️  This will modify your database by adding new columns.")
    print("   A backup will be created automatically.")
    response = input("\n🔄 Proceed with migration? (y/n): ").strip().lower()
    
    if response != 'y':
        logger.info("❌ Migration cancelled by user")
        sys.exit(0)
    
    # Create backup
    logger.info("\n📦 Creating backup...")
    backup_path = backup_database(db_path)
    
    if not backup_path:
        logger.error("❌ Cannot proceed without backup")
        sys.exit(1)
    
    # Run migration
    try:
        changes = migrate_database(db_path)
        
        if changes > 0:
            # Verify migration
            verify_migration(db_path)
            
            print("\n✅ Migration complete!")
            print(f"   Backup saved at: {backup_path}")
            print("\n🚀 You can now use two-temporal detection!")
            print("   Try: python app.py")
            print("   Endpoint: POST /api/detect-change\n")
        else:
            logger.info("\n✅ Database already up-to-date - no changes needed\n")
        
    except Exception as e:
        logger.error(f"\n❌ Migration failed: {e}")
        logger.info(f"🔄 Restore from backup: {backup_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()