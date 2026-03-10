import sqlite3
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from utils.config import DB_PATH


class LocalDBManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._ensure_db_dir()
        self._init_db()

    def _ensure_db_dir(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS real_time_posture (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT DEFAULT 'default',
            detect_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ear_shoulder_angle REAL,
            shoulder_hip_angle REAL,
            anomaly_flag INTEGER DEFAULT 0,
            anomaly_type TEXT,
            spine_angle REAL,
            hunchback_flag INTEGER DEFAULT 0
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT DEFAULT 'default',
            memory_type TEXT NOT NULL,
            memory_content TEXT NOT NULL,
            create_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expire_ts TIMESTAMP
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT DEFAULT 'default',
            high_freq_anomaly TEXT,
            anomaly_time_slot TEXT,
            report_prefer TEXT,
            improve_strategy TEXT,
            update_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS rag_corpus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT DEFAULT 'default',
            corpus_content TEXT NOT NULL,
            create_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tag TEXT
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS rag_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rag_enabled INTEGER DEFAULT 1,
            cloud_enabled INTEGER DEFAULT 0,
            retrieval_strategy TEXT DEFAULT 'local_bm25'
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_reports (
            date TEXT PRIMARY KEY,
            report_content TEXT,
            stats_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            date TEXT PRIMARY KEY,
            good_posture_ratio REAL,
            avg_ear_shoulder REAL,
            avg_shoulder_hip REAL,
            posture_changes INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        cursor.execute('SELECT COUNT(*) FROM rag_config')
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
            INSERT INTO rag_config (rag_enabled, cloud_enabled, retrieval_strategy)
            VALUES (1, 0, 'local_bm25')
            ''')

        conn.commit()
        conn.close()

    def insert_posture_data(self, user_id: str = 'default', **kwargs):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO real_time_posture 
        (user_id, detect_ts, ear_shoulder_angle, shoulder_hip_angle, anomaly_flag, anomaly_type, spine_angle, hunchback_flag)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            kwargs.get('detect_ts', datetime.now()),
            kwargs.get('ear_shoulder_angle'),
            kwargs.get('shoulder_hip_angle'),
            kwargs.get('anomaly_flag', 0),
            kwargs.get('anomaly_type'),
            kwargs.get('spine_angle'),
            kwargs.get('hunchback_flag', 0)
        ))
        conn.commit()
        conn.close()

    def get_posture_data(self, user_id: str = 'default', hours: int = 24):
        conn = self._get_connection()
        cursor = conn.cursor()
        start_time = datetime.now() - timedelta(hours=hours)
        cursor.execute('''
        SELECT * FROM real_time_posture 
        WHERE user_id = ? AND detect_ts >= ?
        ORDER BY detect_ts DESC
        ''', (user_id, start_time))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def add_user_memory(self, user_id: str = 'default', memory_type: str = 'short', 
                       memory_content: str = '', expire_hours: int = 24):
        conn = self._get_connection()
        cursor = conn.cursor()
        expire_ts = None
        if memory_type == 'short' and expire_hours:
            expire_ts = datetime.now() + timedelta(hours=expire_hours)
        cursor.execute('''
        INSERT INTO user_memory (user_id, memory_type, memory_content, expire_ts)
        VALUES (?, ?, ?, ?)
        ''', (user_id, memory_type, memory_content, expire_ts))
        conn.commit()
        conn.close()

    def get_user_memory(self, user_id: str = 'default', memory_type: str = None):
        conn = self._get_connection()
        cursor = conn.cursor()
        if memory_type:
            cursor.execute('''
            SELECT * FROM user_memory 
            WHERE user_id = ? AND memory_type = ? 
            AND (expire_ts IS NULL OR expire_ts > ?)
            ORDER BY create_ts DESC
            ''', (user_id, memory_type, datetime.now()))
        else:
            cursor.execute('''
            SELECT * FROM user_memory 
            WHERE user_id = ? 
            AND (expire_ts IS NULL OR expire_ts > ?)
            ORDER BY create_ts DESC
            ''', (user_id, datetime.now()))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def update_user_profile(self, user_id: str = 'default', **kwargs):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM user_profile WHERE user_id = ?', (user_id,))
        exists = cursor.fetchone()
        if exists:
            cursor.execute('''
            UPDATE user_profile 
            SET high_freq_anomaly = ?, anomaly_time_slot = ?, report_prefer = ?, 
                improve_strategy = ?, update_ts = ?
            WHERE user_id = ?
            ''', (
                kwargs.get('high_freq_anomaly'),
                kwargs.get('anomaly_time_slot'),
                kwargs.get('report_prefer'),
                kwargs.get('improve_strategy'),
                datetime.now(),
                user_id
            ))
        else:
            cursor.execute('''
            INSERT INTO user_profile 
            (user_id, high_freq_anomaly, anomaly_time_slot, report_prefer, improve_strategy)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id,
                kwargs.get('high_freq_anomaly'),
                kwargs.get('anomaly_time_slot'),
                kwargs.get('report_prefer'),
                kwargs.get('improve_strategy')
            ))
        conn.commit()
        conn.close()

    def get_user_profile(self, user_id: str = 'default'):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM user_profile WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def add_rag_corpus(self, user_id: str = 'default', corpus_content: str = '', tag: str = ''):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO rag_corpus (user_id, corpus_content, tag)
        VALUES (?, ?, ?)
        ''', (user_id, corpus_content, tag))
        conn.commit()
        conn.close()

    def get_rag_corpus(self, user_id: str = 'default', tag: str = None):
        conn = self._get_connection()
        cursor = conn.cursor()
        if tag:
            cursor.execute('''
            SELECT * FROM rag_corpus WHERE user_id = ? AND tag = ?
            ORDER BY create_ts DESC
            ''', (user_id, tag))
        else:
            cursor.execute('''
            SELECT * FROM rag_corpus WHERE user_id = ?
            ORDER BY create_ts DESC
            ''', (user_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def save_daily_report(self, date_str: str, report_content: str, stats: Dict):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO daily_reports (date, report_content, stats_json)
        VALUES (?, ?, ?)
        ''', (date_str, report_content, json.dumps(stats)))
        conn.commit()
        conn.close()

    def get_daily_report(self, date_str: str):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM daily_reports WHERE date = ?', (date_str,))
        row = cursor.fetchone()
        conn.close()
        if row:
            result = dict(row)
            result['stats_json'] = json.loads(result['stats_json'])
            return result
        return None

    def save_daily_stats(self, date_str: str, stats: Dict):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO daily_stats 
        (date, good_posture_ratio, avg_ear_shoulder, avg_shoulder_hip, posture_changes)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            date_str,
            stats.get('good_posture_ratio', 0),
            stats.get('avg_ear_shoulder', 0),
            stats.get('avg_shoulder_hip', 0),
            stats.get('posture_changes', 0)
        ))
        conn.commit()
        conn.close()

    def get_daily_stats(self, date_str: str):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM daily_stats WHERE date = ?', (date_str,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_trend_stats(self, days: int = 7):
        conn = self._get_connection()
        cursor = conn.cursor()
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        cursor.execute('''
        SELECT date, good_posture_ratio 
        FROM daily_stats 
        WHERE date BETWEEN ? AND ?
        ORDER BY date
        ''', (start_date.isoformat(), end_date.isoformat()))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def clean_expired_data(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        DELETE FROM user_memory WHERE expire_ts IS NOT NULL AND expire_ts < ?
        ''', (datetime.now(),))
        cutoff = datetime.now() - timedelta(days=90)
        cursor.execute('''
        DELETE FROM real_time_posture WHERE detect_ts < ?
        ''', (cutoff,))
        conn.commit()
        conn.close()


db_manager = LocalDBManager()
