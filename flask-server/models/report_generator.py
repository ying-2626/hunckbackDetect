import pandas as pd
import os
import json
import sqlite3
from .config import LOG_FILE
from .prompts import (
    REPORT_GENERATION_SYSTEM_PROMPT,
    REPORT_GENERATION_WITH_RAG_PROMPT,
    REPORT_GENERATION_WITH_HISTORY_PROMPT
)
from openai import OpenAI
from datetime import datetime, timedelta

def load_config():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        d = current_dir
        while os.path.dirname(d) != d:
            config_path = os.path.join(d, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            d = os.path.dirname(d)
    except Exception as e:
        print(f"Warning: Failed to load config.json: {e}")
    return {}

CONFIG = load_config()

OPENAI_API_KEY = CONFIG.get("DASHSCOPE_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: 'DASHSCOPE_API_KEY' not found in config.json.")
    pass

OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

try:
    OPENAI_CLIENT = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    OPENAI_CLIENT = None


class ReportGenerator:
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports.db')
        self._init_db()
        self.rag_service = None
    
    def _init_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
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
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database initialization failed: {e}")
    
    def _get_rag_service(self):
        if self.rag_service is None:
            try:
                from flask import has_request_context, current_app
                if has_request_context():
                    self.rag_service = current_app.extensions['hunchback'].get('rag_service')
            except:
                pass
        return self.rag_service
    
    def _search_rag_knowledge(self, query: str, top_k: int = 3):
        rag = self._get_rag_service()
        if rag:
            try:
                results = rag.search_knowledge(query, top_k=top_k)
                return "\n".join([f"- {r['content']}" for r in results])
            except Exception as e:
                print(f"RAG search failed: {e}")
        return None
    
    def _get_user_trend(self, days: int = 7):
        try:
            conn = sqlite3.connect(self.db_path)
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
            
            if len(rows) &lt; 2:
                return None
            
            ratios = [r[1] for r in rows]
            avg_ratio = sum(ratios) / len(ratios)
            first_ratio = ratios[0]
            last_ratio = ratios[-1]
            
            if last_ratio &gt; first_ratio + 5:
                trend = "improving"
            elif last_ratio &lt; first_ratio - 5:
                trend = "declining"
            else:
                trend = "stable"
            
            return {
                "days": days,
                "avg_good_ratio": avg_ratio,
                "trend": trend,
                "first_ratio": first_ratio,
                "last_ratio": last_ratio
            }
        except Exception as e:
            print(f"Get trend failed: {e}")
            return None
    
    def _save_daily_stats(self, date_str: str, stats: dict):
        try:
            conn = sqlite3.connect(self.db_path)
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
        except Exception as e:
            print(f"Save daily stats failed: {e}")

    def generate_report(self):
        try:
            df = pd.read_csv(LOG_FILE)
            logs = df.tail(20).to_string(index=False)

            completion = OPENAI_CLIENT.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": REPORT_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": "这是今天的监测日志数据，请生成报告：\n" + logs},
                ],
            )
            return completion.choices[0].message.content

        except Exception as e:
            return f"生成报告失败: {str(e)}"

    def generate_daily_report(self, date_str):
        try:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT report_content, stats_json FROM daily_reports WHERE date = ?", (date_str,))
                row = cursor.fetchone()
                conn.close()

                if row:
                    print(f"从数据库加载 {date_str} 的报告")
                    return {
                        "report": row[0],
                        "stats": json.loads(row[1]),
                        "date": date_str
                    }
            except Exception as e:
                print(f"读取数据库失败，尝试重新生成: {e}")

            df = pd.read_csv(LOG_FILE)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            target_date = pd.to_datetime(date_str).date()
            daily_df = df[df['Timestamp'].dt.date == target_date].copy()

            if daily_df.empty:
                return {
                    "report": f"{date_str} 无坐姿监测数据",
                    "stats": {},
                    "date": date_str
                }

            def to_bool(val):
                if isinstance(val, bool):
                    return val
                if isinstance(val, (int, float)):
                    return bool(val)
                if isinstance(val, str):
                    return val.strip().lower() == 'true'
                return False

            daily_df['PostureStatus'] = daily_df['PostureStatus'].apply(to_bool)
            total = ((daily_df['PostureStatus'] == False) | (daily_df['PostureStatus'] == True)).sum()
            good = (daily_df['PostureStatus'] == False).sum()
            good_posture_ratio = (good / total * 100) if total &gt; 0 else 0

            stats = {
                "avg_ear_shoulder": float(daily_df['EarShoulderDiff'].mean()),
                "avg_shoulder_hip": float(daily_df['ShoulderHipDiff'].mean()),
                "good_posture_ratio": float(good_posture_ratio),
                "posture_changes": int(len(daily_df)),
                "max_ear_shoulder": float(daily_df['EarShoulderDiff'].max()),
                "min_shoulder_hip": float(daily_df['ShoulderHipDiff'].min()),
            }

            self._save_daily_stats(date_str, stats)

            system_prompt = REPORT_GENERATION_SYSTEM_PROMPT
            
            rag_context = None
            if good_posture_ratio &lt; 70:
                rag_context = self._search_rag_knowledge(
                    f"不良坐姿比例 {good_posture_ratio:.1f}% 如何改善坐姿"
                )
            elif good_posture_ratio &lt; 85:
                rag_context = self._search_rag_knowledge("如何保持正确坐姿")
            
            if rag_context:
                system_prompt = REPORT_GENERATION_WITH_RAG_PROMPT.format(
                    base_prompt=system_prompt,
                    rag_context=rag_context
                )

            history_trend = self._get_user_trend(days=7)
            if history_trend:
                history_context = f"过去7天平均良好坐姿比例: {history_trend['avg_good_ratio']:.1f}%\n"
                history_context += f"趋势: {history_trend['trend']}\n"
                history_context += f"起始比例: {history_trend['first_ratio']:.1f}%\n"
                history_context += f"最新比例: {history_trend['last_ratio']:.1f}%"
                system_prompt = REPORT_GENERATION_WITH_HISTORY_PROMPT.format(
                    base_prompt=system_prompt,
                    history_context=history_context
                )

            logs = daily_df.to_string(index=False)
            completion = OPENAI_CLIENT.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "请根据以下日志生成一份关于坐姿异常的报告，不要使用markdown语法，不要用头-肩垂直差等专业术语，应该用更通俗地表达使用户可以理解" + logs},
                ],
            )
            report = completion.choices[0].message.content

            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO daily_reports (date, report_content, stats_json) VALUES (?, ?, ?)",
                    (date_str, report, json.dumps(stats))
                )
                conn.commit()
                conn.close()
                print(f"已保存 {date_str} 的报告到数据库")
            except Exception as e:
                print(f"保存数据库失败: {e}")

            return {
                "report": report,
                "stats": stats,
                "date": date_str
            }

        except Exception as e:
            return {
                "report": f"报告生成失败: {str(e)}",
                "stats": {},
                "date": date_str
            }
