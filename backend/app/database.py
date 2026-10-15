import sqlite3
from .config import DATABASE_PATH

def connect():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize():
    with connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS sessions (
          id TEXT PRIMARY KEY, filename TEXT NOT NULL, original_path TEXT NOT NULL,
          working_path TEXT NOT NULL, created_at TEXT NOT NULL, rows INTEGER, columns INTEGER
        );
        CREATE TABLE IF NOT EXISTS history (
          id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, created_at TEXT NOT NULL,
          operation TEXT NOT NULL, parameters TEXT NOT NULL, status TEXT NOT NULL,
          reason TEXT, affected_rows INTEGER DEFAULT 0, affected_cells INTEGER DEFAULT 0,
          before_metrics TEXT, after_metrics TEXT, snapshot_path TEXT
        );''')
