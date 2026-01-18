import sqlite3
import json
import os
import datetime
import numpy as np
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sitting_watch.db')

class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """初始化数据库表结构"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # 1. 检测记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            result_json TEXT,
            class_name TEXT,
            confidence REAL,
            image_path TEXT,
            is_reviewed BOOLEAN DEFAULT 0
        )
        ''')

        # 2. 知识库表 (用于 RAG)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            category TEXT,
            embedding BLOB
        )
        ''')

        # 3. 用户行为摘要表 (用于 RAG)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_date DATE,
            end_date DATE,
            summary_text TEXT,
            embedding BLOB
        )
        ''')

        conn.commit()
        conn.close()

    def add_record(self, result: Dict[str, Any], image_path: str):
        """添加一条检测记录"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 解析 result
        # 假设 result 格式: {"class": "sitting_bad", "conf": "0.76"}
        class_name = result.get('class', 'unknown')
        try:
            confidence = float(result.get('conf', 0.0))
        except:
            confidence = 0.0
            
        cursor.execute('''
        INSERT INTO detection_records (result_json, class_name, confidence, image_path, timestamp)
        VALUES (?, ?, ?, ?, ?)
        ''', (json.dumps(result), class_name, confidence, image_path, datetime.datetime.now()))
        
        conn.commit()
        conn.close()

    def get_records(self, limit: int = 50, offset: int = 0) -> List[dict]:
        """获取历史检测记录"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM detection_records ORDER BY timestamp DESC LIMIT ? OFFSET ?', (limit, offset))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def add_knowledge(self, content: str, category: str, embedding: List[float]):
        """添加知识库条目"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 将 embedding 转换为 bytes 存储
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
        
        cursor.execute('''
        INSERT INTO knowledge_base (content, category, embedding)
        VALUES (?, ?, ?)
        ''', (content, category, embedding_blob))
        
        conn.commit()
        conn.close()

    def add_summary(self, start_date: str, end_date: str, summary_text: str, embedding: List[float]):
        """添加用户行为摘要"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
        
        cursor.execute('''
        INSERT INTO user_summaries (start_date, end_date, summary_text, embedding)
        VALUES (?, ?, ?, ?)
        ''', (start_date, end_date, summary_text, embedding_blob))
        
        conn.commit()
        conn.close()

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def search_knowledge(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """简单的向量检索 (基于内存计算余弦相似度)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, content, category, embedding FROM knowledge_base')
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        results = []

        for row in rows:
            if not row['embedding']:
                continue
            # 从 BLOB 恢复向量
            db_vec = np.frombuffer(row['embedding'], dtype=np.float32)
            score = self._cosine_similarity(query_vec, db_vec)
            results.append({
                'id': row['id'],
                'content': row['content'],
                'category': row['category'],
                'score': score
            })

        # 按相似度降序排列
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

# 全局单例
db = DatabaseManager()
