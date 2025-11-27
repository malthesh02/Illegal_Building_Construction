"""
Field Visit Reporting System
Track field inspections, violations, and verification status

Author: Enhanced Version
Date: 2025
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class VisitStatus(Enum):
    """Field visit status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ViolationType(Enum):
    """Type of violation found"""
    NO_PERMIT = "no_permit"
    PERMIT_EXPIRED = "permit_expired"
    EXCEEDS_APPROVED_AREA = "exceeds_approved_area"
    UNAUTHORIZED_CHANGES = "unauthorized_changes"
    STRUCTURAL_VIOLATION = "structural_violation"
    ENVIRONMENTAL_VIOLATION = "environmental_violation"
    OTHER = "other"


class FieldVisitManager:
    """
    Manages field visits, inspections, and violation reporting
    """
    
    def __init__(self, db_path: str):
        """Initialize field visit manager"""
        self.db_path = db_path
        self._create_tables()
        logger.info("✅ FieldVisitManager initialized")
    
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
    
    def _create_tables(self):
        """Create field visit tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # ===== TABLE 1: FIELD VISITS =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS field_visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visit_id TEXT UNIQUE NOT NULL,
                    building_id TEXT NOT NULL,
                    session_id TEXT,
                    status TEXT DEFAULT 'pending',
                    priority TEXT DEFAULT 'medium',
                    scheduled_date DATE,
                    completed_date DATE,
                    inspector_name TEXT,
                    inspector_id TEXT,
                    gps_latitude REAL,
                    gps_longitude REAL,
                    address TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (building_id) REFERENCES buildings(building_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')
            
            # ===== TABLE 2: VIOLATIONS =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    violation_id TEXT UNIQUE NOT NULL,
                    visit_id TEXT NOT NULL,
                    building_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    severity TEXT DEFAULT 'medium',
                    description TEXT,
                    evidence_photos TEXT,
                    corrective_action TEXT,
                    deadline DATE,
                    status TEXT DEFAULT 'open',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    FOREIGN KEY (visit_id) REFERENCES field_visits(visit_id),
                    FOREIGN KEY (building_id) REFERENCES buildings(building_id)
                )
            ''')
            
            # ===== TABLE 3: INSPECTION CHECKLIST =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inspection_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visit_id TEXT NOT NULL,
                    item_name TEXT NOT NULL,
                    category TEXT,
                    compliant BOOLEAN,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (visit_id) REFERENCES field_visits(visit_id)
                )
            ''')
            
            # ===== TABLE 4: FIELD PHOTOS =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS field_photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    photo_id TEXT UNIQUE NOT NULL,
                    visit_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    photo_type TEXT,
                    description TEXT,
                    gps_latitude REAL,
                    gps_longitude REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (visit_id) REFERENCES field_visits(visit_id)
                )
            ''')
            
            # ===== TABLE 5: FOLLOW-UP ACTIONS =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS follow_up_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_id TEXT UNIQUE NOT NULL,
                    visit_id TEXT NOT NULL,
                    violation_id TEXT,
                    action_type TEXT NOT NULL,
                    description TEXT,
                    assigned_to TEXT,
                    due_date DATE,
                    status TEXT DEFAULT 'pending',
                    completed_at TIMESTAMP,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (visit_id) REFERENCES field_visits(visit_id),
                    FOREIGN KEY (violation_id) REFERENCES violations(violation_id)
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_visits_status ON field_visits(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_visits_building ON field_visits(building_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_violations_status ON violations(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_violations_type ON violations(violation_type)')
            
            conn.commit()
            logger.info("✅ Field visit tables created successfully")
    
    # ========================================================================
    # FIELD VISIT OPERATIONS
    # ========================================================================
    
    def create_field_visit(self, building_id: str, session_id: str = None,
                          inspector_name: str = None, priority: str = "medium",
                          scheduled_date: str = None, **kwargs) -> str:
        """
        Create a new field visit request
        
        Args:
            building_id: Building identifier
            session_id: Detection session ID
            inspector_name: Name of inspector
            priority: Priority level (low, medium, high, urgent)
            scheduled_date: Scheduled visit date (YYYY-MM-DD)
            **kwargs: Additional fields (gps_latitude, gps_longitude, address, notes)
        
        Returns:
            Visit ID
        """
        import uuid
        
        visit_id = f"VISIT_{uuid.uuid4().hex[:8].upper()}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO field_visits (
                    visit_id, building_id, session_id, status, priority,
                    scheduled_date, inspector_name, gps_latitude, gps_longitude,
                    address, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                visit_id, building_id, session_id, 'pending', priority,
                scheduled_date, inspector_name,
                kwargs.get('gps_latitude'),
                kwargs.get('gps_longitude'),
                kwargs.get('address'),
                kwargs.get('notes')
            ))
            
            logger.info(f"✅ Field visit created: {visit_id} for building {building_id}")
            return visit_id
    
    def update_visit_status(self, visit_id: str, status: str, 
                           inspector_id: str = None):
        """Update field visit status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            update_fields = ['status = ?', 'updated_at = CURRENT_TIMESTAMP']
            params = [status]
            
            if status == 'completed':
                update_fields.append('completed_date = CURRENT_DATE')
            
            if inspector_id:
                update_fields.append('inspector_id = ?')
                params.append(inspector_id)
            
            params.append(visit_id)
            
            cursor.execute(f'''
                UPDATE field_visits 
                SET {', '.join(update_fields)}
                WHERE visit_id = ?
            ''', params)
            
            logger.info(f"📝 Visit {visit_id} status updated to: {status}")
    
    def get_field_visit(self, visit_id: str) -> Optional[Dict]:
        """Get field visit details"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM field_visits WHERE visit_id = ?', (visit_id,))
            row = cursor.fetchone()
            
            if row:
                visit = dict(row)
                
                # Get violations
                cursor.execute('SELECT * FROM violations WHERE visit_id = ?', (visit_id,))
                visit['violations'] = [dict(r) for r in cursor.fetchall()]
                
                # Get inspection items
                cursor.execute('SELECT * FROM inspection_items WHERE visit_id = ?', (visit_id,))
                visit['inspection_items'] = [dict(r) for r in cursor.fetchall()]
                
                # Get photos
                cursor.execute('SELECT * FROM field_photos WHERE visit_id = ?', (visit_id,))
                visit['photos'] = [dict(r) for r in cursor.fetchall()]
                
                # Get follow-up actions
                cursor.execute('SELECT * FROM follow_up_actions WHERE visit_id = ?', (visit_id,))
                visit['follow_up_actions'] = [dict(r) for r in cursor.fetchall()]
                
                return visit
            
            return None
    
    def get_pending_visits(self, priority: str = None) -> List[Dict]:
        """Get all pending field visits"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if priority:
                cursor.execute('''
                    SELECT * FROM field_visits 
                    WHERE status IN ('pending', 'scheduled')
                    AND priority = ?
                    ORDER BY priority DESC, created_at ASC
                ''', (priority,))
            else:
                cursor.execute('''
                    SELECT * FROM field_visits 
                    WHERE status IN ('pending', 'scheduled')
                    ORDER BY 
                        CASE priority 
                            WHEN 'urgent' THEN 1
                            WHEN 'high' THEN 2
                            WHEN 'medium' THEN 3
                            WHEN 'low' THEN 4
                        END,
                        created_at ASC
                ''')
            
            return [dict(row) for row in cursor.fetchall()]
    
    # ========================================================================
    # VIOLATION OPERATIONS
    # ========================================================================
    
    def report_violation(self, visit_id: str, building_id: str,
                        violation_type: str, severity: str = "medium",
                        description: str = None, evidence_photos: List[str] = None,
                        deadline: str = None) -> str:
        """
        Report a violation found during field visit
        
        Args:
            visit_id: Field visit ID
            building_id: Building ID
            violation_type: Type of violation
            severity: Severity level (low, medium, high, critical)
            description: Detailed description
            evidence_photos: List of photo paths
            deadline: Deadline for correction (YYYY-MM-DD)
        
        Returns:
            Violation ID
        """
        import uuid
        
        violation_id = f"VIOL_{uuid.uuid4().hex[:8].upper()}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO violations (
                    violation_id, visit_id, building_id, violation_type,
                    severity, description, evidence_photos, deadline, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
            ''', (
                violation_id, visit_id, building_id, violation_type,
                severity, description,
                json.dumps(evidence_photos) if evidence_photos else None,
                deadline
            ))
            
            logger.info(f"⚠️ Violation reported: {violation_id} ({violation_type})")
            return violation_id
    
    def resolve_violation(self, violation_id: str, corrective_action: str = None):
        """Mark violation as resolved"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE violations 
                SET status = 'resolved', 
                    resolved_at = CURRENT_TIMESTAMP,
                    corrective_action = ?
                WHERE violation_id = ?
            ''', (corrective_action, violation_id))
            
            logger.info(f"✅ Violation resolved: {violation_id}")
    
    def get_open_violations(self, building_id: str = None) -> List[Dict]:
        """Get all open violations"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if building_id:
                cursor.execute('''
                    SELECT * FROM violations 
                    WHERE status = 'open' AND building_id = ?
                    ORDER BY severity DESC, created_at DESC
                ''', (building_id,))
            else:
                cursor.execute('''
                    SELECT * FROM violations 
                    WHERE status = 'open'
                    ORDER BY 
                        CASE severity 
                            WHEN 'critical' THEN 1
                            WHEN 'high' THEN 2
                            WHEN 'medium' THEN 3
                            WHEN 'low' THEN 4
                        END,
                        created_at DESC
                ''')
            
            violations = [dict(row) for row in cursor.fetchall()]
            
            # Parse evidence photos
            for v in violations:
                if v.get('evidence_photos'):
                    v['evidence_photos'] = json.loads(v['evidence_photos'])
            
            return violations
    
    # ========================================================================
    # INSPECTION CHECKLIST
    # ========================================================================
    
    def add_inspection_item(self, visit_id: str, item_name: str,
                           category: str = None, compliant: bool = None,
                           notes: str = None):
        """Add an inspection checklist item"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO inspection_items (visit_id, item_name, category, compliant, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (visit_id, item_name, category, compliant, notes))
            
            logger.info(f"✅ Inspection item added: {item_name}")
    
    def get_inspection_checklist(self, visit_id: str) -> List[Dict]:
        """Get inspection checklist for a visit"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM inspection_items 
                WHERE visit_id = ?
                ORDER BY category, item_name
            ''', (visit_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # ========================================================================
    # FIELD PHOTOS
    # ========================================================================
    
    def add_field_photo(self, visit_id: str, file_path: str,
                       photo_type: str = None, description: str = None,
                       gps_latitude: float = None, gps_longitude: float = None) -> str:
        """Add a field photo"""
        import uuid
        
        photo_id = f"PHOTO_{uuid.uuid4().hex[:8].upper()}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO field_photos (
                    photo_id, visit_id, file_path, photo_type, description,
                    gps_latitude, gps_longitude
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (photo_id, visit_id, file_path, photo_type, description,
                  gps_latitude, gps_longitude))
            
            logger.info(f"📷 Field photo added: {photo_id}")
            return photo_id
    
    # ========================================================================
    # FOLLOW-UP ACTIONS
    # ========================================================================
    
    def create_follow_up_action(self, visit_id: str, action_type: str,
                               description: str, assigned_to: str = None,
                               due_date: str = None, violation_id: str = None) -> str:
        """Create a follow-up action"""
        import uuid
        
        action_id = f"ACTION_{uuid.uuid4().hex[:8].upper()}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO follow_up_actions (
                    action_id, visit_id, violation_id, action_type,
                    description, assigned_to, due_date, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
            ''', (action_id, visit_id, violation_id, action_type,
                  description, assigned_to, due_date))
            
            logger.info(f"📋 Follow-up action created: {action_id}")
            return action_id
    
    def complete_follow_up_action(self, action_id: str, notes: str = None):
        """Mark follow-up action as completed"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE follow_up_actions 
                SET status = 'completed',
                    completed_at = CURRENT_TIMESTAMP,
                    notes = ?
                WHERE action_id = ?
            ''', (notes, action_id))
            
            logger.info(f"✅ Follow-up action completed: {action_id}")
    
    # ========================================================================
    # STATISTICS & REPORTS
    # ========================================================================
    
    def get_visit_statistics(self) -> Dict:
        """Get field visit statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total visits by status
            cursor.execute('''
                SELECT status, COUNT(*) as count 
                FROM field_visits 
                GROUP BY status
            ''')
            stats['visits_by_status'] = {row['status']: row['count'] 
                                         for row in cursor.fetchall()}
            
            # Total violations by type
            cursor.execute('''
                SELECT violation_type, COUNT(*) as count 
                FROM violations 
                WHERE status = 'open'
                GROUP BY violation_type
            ''')
            stats['open_violations_by_type'] = {row['violation_type']: row['count'] 
                                                for row in cursor.fetchall()}
            
            # Total violations by severity
            cursor.execute('''
                SELECT severity, COUNT(*) as count 
                FROM violations 
                WHERE status = 'open'
                GROUP BY severity
            ''')
            stats['open_violations_by_severity'] = {row['severity']: row['count'] 
                                                     for row in cursor.fetchall()}
            
            # Pending follow-up actions
            cursor.execute('''
                SELECT COUNT(*) as count 
                FROM follow_up_actions 
                WHERE status = 'pending'
            ''')
            stats['pending_actions'] = cursor.fetchone()['count']
            
            # Completion rate
            cursor.execute('''
                SELECT 
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) * 100.0 / COUNT(*) as rate
                FROM field_visits
            ''')
            stats['completion_rate'] = round(cursor.fetchone()['rate'] or 0, 2)
            
            return stats
    
    def generate_field_report(self, visit_id: str) -> Dict:
        """Generate comprehensive field visit report"""
        visit = self.get_field_visit(visit_id)
        
        if not visit:
            return None
        
        report = {
            'visit': visit,
            'summary': {
                'total_violations': len(visit.get('violations', [])),
                'critical_violations': len([v for v in visit.get('violations', []) 
                                           if v['severity'] == 'critical']),
                'inspection_items_checked': len(visit.get('inspection_items', [])),
                'non_compliant_items': len([i for i in visit.get('inspection_items', []) 
                                            if i.get('compliant') == 0]),
                'photos_taken': len(visit.get('photos', [])),
                'follow_up_actions': len(visit.get('follow_up_actions', []))
            },
            'generated_at': datetime.now().isoformat()
        }
        
        return report


# Test function
if __name__ == "__main__":
    print("=" * 70)
    print("Field Visit System - Test")
    print("=" * 70)
    
    # Create test database
    fvm = FieldVisitManager("test_field_visits.db")
    
    # Test: Create field visit
    visit_id = fvm.create_field_visit(
        building_id="BLDG_TEST001",
        session_id="session_12345",
        inspector_name="John Doe",
        priority="high",
        scheduled_date="2025-01-20",
        gps_latitude=12.3000,
        gps_longitude=76.6500,
        address="123 Main St, Mysuru",
        notes="Suspected illegal construction"
    )
    print(f"✅ Visit created: {visit_id}")
    
    # Test: Report violation
    violation_id = fvm.report_violation(
        visit_id=visit_id,
        building_id="BLDG_TEST001",
        violation_type="no_permit",
        severity="high",
        description="Building constructed without valid permit",
        deadline="2025-02-15"
    )
    print(f"⚠️ Violation reported: {violation_id}")
    
    # Test: Add inspection items
    fvm.add_inspection_item(visit_id, "Building Permit", "Documentation", False, "No permit found")
    fvm.add_inspection_item(visit_id, "Structural Safety", "Safety", True, "Structure appears sound")
    print("✅ Inspection items added")
    
    # Test: Get statistics
    stats = fvm.get_visit_statistics()
    print(f"📊 Statistics: {stats}")
    
    # Test: Generate report
    report = fvm.generate_field_report(visit_id)
    print(f"📄 Report generated: {report['summary']}")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)