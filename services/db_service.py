import sqlite3

import pandas as pd

from config.settings import DB_PATH, DEFAULT_CAMERAS, VIOLATION_ALERT_THRESHOLD


def get_db_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the SQLite database schema."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create Violations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        location TEXT NOT NULL,
        violation_type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        plate_number TEXT NOT NULL,
        confidence REAL NOT NULL,
        evidence_path TEXT
    )
    """)

    # Create Cameras table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cameras (
        camera_id TEXT PRIMARY KEY,
        location TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL
    )
    """)

    # Create Alerts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_number TEXT UNIQUE NOT NULL,
        violation_count INTEGER DEFAULT 1,
        last_violation_type TEXT,
        timestamp TEXT NOT NULL,
        details TEXT
    )
    """)

    # Populate default cameras if table is empty
    cursor.execute("SELECT COUNT(*) FROM cameras")
    if cursor.fetchone()[0] == 0:
        for cam in DEFAULT_CAMERAS:
            cursor.execute("""
            INSERT INTO cameras (camera_id, location, latitude, longitude)
            VALUES (?, ?, ?, ?)
            """, (cam["id"], cam["location"], cam["latitude"], cam["longitude"]))

    conn.commit()
    conn.close()


def add_camera_if_not_exists(camera_id, location, latitude, longitude):
    """Inserts a camera into the cameras table if it doesn't already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO cameras (camera_id, location, latitude, longitude)
    VALUES (?, ?, ?, ?)
    """, (camera_id, location, latitude, longitude))
    conn.commit()
    conn.close()


def add_violation(camera_id, location, violation_type, timestamp, plate_number, confidence, evidence_path):
    """Logs a violation in the SQLite database and triggers repeat offender alerts if applicable."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Add violation
    cursor.execute("""
    INSERT INTO violations (camera_id, location, violation_type, timestamp, plate_number, confidence, evidence_path)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (camera_id, location, violation_type, timestamp, plate_number, confidence, evidence_path))

    # Check for repeat offender
    if plate_number and plate_number != "UNKNOWN":
        cursor.execute("SELECT COUNT(*) FROM violations WHERE plate_number = ?", (plate_number,))
        count = cursor.fetchone()[0]

        if count >= VIOLATION_ALERT_THRESHOLD:
            # We have a repeat offender, insert or update alerts table
            cursor.execute("SELECT MIN(timestamp) FROM violations WHERE plate_number = ?", (plate_number,))
            first_seen = cursor.fetchone()[0]
            
            # Fetch all violations for details
            cursor.execute("SELECT violation_type, timestamp, location FROM violations WHERE plate_number = ? ORDER BY timestamp DESC", (plate_number,))
            records = cursor.fetchall()
            details_str = "; ".join([f"{r[0]} at {r[2]} ({r[1]})" for r in records])

            cursor.execute("""
            INSERT INTO alerts (plate_number, violation_count, last_violation_type, timestamp, details)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(plate_number) DO UPDATE SET
                violation_count = excluded.violation_count,
                last_violation_type = excluded.last_violation_type,
                timestamp = excluded.timestamp,
                details = excluded.details
            """, (plate_number, count, violation_type, timestamp, details_str))

    conn.commit()
    conn.close()


def get_recent_violations(limit=100, date_prefix=None):
    """Fetches the most recent violations as a pandas DataFrame."""
    conn = get_db_connection()
    sql = """
    SELECT timestamp as Time, camera_id as 'Camera ID', location as Location,
           violation_type as 'Violation Type', plate_number as 'Plate Number',
           ROUND(confidence, 2) as Confidence, evidence_path as Evidence
    FROM violations
    """
    params = []
    if date_prefix:
        sql += " WHERE timestamp LIKE ?"
        params.append(f"{date_prefix}%")
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def search_violations(query="", violation_type=None, camera_id=None, limit=1000, date_prefix=None):
    """Search violations using text filters and an optional violation type filter."""
    conn = get_db_connection()
    sql = """
    SELECT id, timestamp as Time, camera_id as 'Camera ID', location as Location,
           violation_type as 'Violation Type', plate_number as 'Plate Number',
           ROUND(confidence, 2) as Confidence, evidence_path as Evidence
    FROM violations
    WHERE 1=1
    """
    params = []
    if query:
        sql += " AND (plate_number LIKE ? OR location LIKE ? OR camera_id LIKE ?)"
        wildcard = f"%{query}%"
        params.extend([wildcard, wildcard, wildcard])
    if violation_type and violation_type != "ALL":
        sql += " AND violation_type = ?"
        params.append(violation_type)
    if camera_id and camera_id != "ALL":
        sql += " AND camera_id = ?"
        params.append(camera_id)
    if date_prefix:
        sql += " AND timestamp LIKE ?"
        params.append(f"{date_prefix}%")
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def get_all_violations():
    """Fetches all violations as a list of dictionaries."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM violations ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_violations_by_type(date_prefix=None):
    """Gets counts of violations grouped by type."""
    conn = get_db_connection()
    sql = """
    SELECT violation_type as 'Violation Type', COUNT(*) as Count
    FROM violations
    """
    params = []
    if date_prefix:
        sql += " WHERE timestamp LIKE ?"
        params.append(f"{date_prefix}%")
    sql += " GROUP BY violation_type"
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def get_violations_over_time(date_prefix=None):
    """Gets counts of violations grouped by time/date."""
    conn = get_db_connection()
    sql = """
    SELECT SUBSTR(timestamp, 1, 10) as Date, COUNT(*) as Count
    FROM violations
    """
    params = []
    if date_prefix:
        sql += " WHERE timestamp LIKE ?"
        params.append(f"{date_prefix}%")
    sql += " GROUP BY Date ORDER BY Date ASC"
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def get_camera_wise_violations(date_prefix=None):
    """Gets violation counts grouped by camera location."""
    conn = get_db_connection()
    sql = """
    SELECT location as Location, COUNT(*) as Count
    FROM violations
    """
    params = []
    if date_prefix:
        sql += " WHERE timestamp LIKE ?"
        params.append(f"{date_prefix}%")
    sql += " GROUP BY location"
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def get_repeat_offender_alerts(date_prefix=None):
    """Fetches repeat offender alerts.

    If date_prefix is provided, only alerts whose latest timestamp begins with
    that prefix are returned. This keeps the dashboard aligned with the
    "today" style wording when needed.
    """
    conn = get_db_connection()
    query = """
    SELECT plate_number as 'Plate Number', violation_count as 'Offense Count', 
           last_violation_type as 'Latest Violation', timestamp as 'Last Detected',
           details as 'Offense History'
    FROM alerts
    """
    params = ()
    if date_prefix:
        query += " WHERE timestamp LIKE ?"
        params = (f"{date_prefix}%",)
    query += " ORDER BY violation_count DESC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_cameras_with_density(date_prefix=None):
    """Gets camera information combined with total violations for heatmap density mapping."""
    conn = get_db_connection()
    sql = """
    SELECT c.camera_id, c.location, c.latitude, c.longitude, COALESCE(v.count, 0) as count
    FROM cameras c
    LEFT JOIN (
        SELECT camera_id, COUNT(*) as count
        FROM violations
    """
    params = []
    if date_prefix:
        sql += " WHERE timestamp LIKE ?"
        params.append(f"{date_prefix}%")
    sql += " GROUP BY camera_id"
    sql += ") v ON c.camera_id = v.camera_id"
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def get_counts_summary(date_prefix=None):
    """Returns a dictionary with counts for the dashboard metric cards."""
    conn = get_db_connection()
    cursor = conn.cursor()

    if date_prefix:
        cursor.execute("SELECT COUNT(*) FROM violations WHERE timestamp LIKE ?", (f"{date_prefix}%",))
    else:
        cursor.execute("SELECT COUNT(*) FROM violations")
    total = cursor.fetchone()[0]

    if date_prefix:
        cursor.execute(
            "SELECT COUNT(*) FROM violations WHERE violation_type = 'Helmet Violation' AND timestamp LIKE ?",
            (f"{date_prefix}%",),
        )
    else:
        cursor.execute("SELECT COUNT(*) FROM violations WHERE violation_type = 'Helmet Violation'")
    helmet = cursor.fetchone()[0]

    if date_prefix:
        cursor.execute(
            "SELECT COUNT(*) FROM violations WHERE violation_type = 'Triple Riding' AND timestamp LIKE ?",
            (f"{date_prefix}%",),
        )
    else:
        cursor.execute("SELECT COUNT(*) FROM violations WHERE violation_type = 'Triple Riding'")
    triple = cursor.fetchone()[0]

    if date_prefix:
        cursor.execute(
            "SELECT COUNT(*) FROM violations WHERE violation_type = 'Illegal Parking' AND timestamp LIKE ?",
            (f"{date_prefix}%",),
        )
    else:
        cursor.execute("SELECT COUNT(*) FROM violations WHERE violation_type = 'Illegal Parking'")
    parking = cursor.fetchone()[0]

    if date_prefix:
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE timestamp LIKE ?", (f"{date_prefix}%",))
    else:
        cursor.execute("SELECT COUNT(*) FROM alerts")
    repeat = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM cameras")
    c_count = cursor.fetchone()[0]

    conn.close()

    return {
        "total": total,
        "helmet": helmet,
        "triple": triple,
        "parking": parking,
        "repeat": repeat,
        "cameras": c_count
    }


def get_alerts_count():
    """Return the current number of repeat-offender alerts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM alerts")
    count = cursor.fetchone()[0]
    conn.close()
    return count
