import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.local_db_manager import db_manager
from utils.config import (
    SHORT_TERM_MEMORY_EXPIRE_HOURS,
    LONG_TERM_MEMORY_DAYS,
    PROFILE_UPDATE_INTERVAL_DAYS
)


class LightProfileManager:
    def __init__(self):
        self.db = db_manager

    def generate_short_term_memory(self, user_id: str = 'default'):
        posture_data = self.db.get_posture_data(user_id, hours=24)
        
        if not posture_data:
            return None
        
        anomaly_count = sum(1 for p in posture_data if p['anomaly_flag'])
        hunchback_count = sum(1 for p in posture_data if p['hunchback_flag'])
        total_count = len(posture_data)
        
        anomaly_slots = defaultdict(int)
        for p in posture_data:
            if p['anomaly_flag'] or p['hunchback_flag']:
                ts = p['detect_ts']
                if isinstance(ts, str):
                    try:
                        ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
                    except:
                        try:
                            ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                        except:
                            continue
                hour_slot = ts.hour
                anomaly_slots[hour_slot] += 1
        
        high_freq_slot = None
        if anomaly_slots:
            high_freq_slot = max(anomaly_slots.items(), key=lambda x: x[1])[0]
        
        memory_content = f"近24小时检测{total_count}次，异常{anomaly_count}次，驼背{hunchback_count}次。"
        if high_freq_slot is not None:
            memory_content += f"高频异常时段:{high_freq_slot}:00-{high_freq_slot+1}:00。"
        
        self.db.add_user_memory(
            user_id=user_id,
            memory_type='short',
            memory_content=memory_content,
            expire_hours=SHORT_TERM_MEMORY_EXPIRE_HOURS
        )
        
        return memory_content

    def generate_long_term_memory(self, user_id: str = 'default'):
        short_memories = self.db.get_user_memory(user_id, memory_type='short')
        
        if not short_memories:
            return None
        
        total_anomalies = 0
        total_detections = 0
        slot_counts = defaultdict(int)
        
        for mem in short_memories:
            content = mem['memory_content']
            if '检测' in content and '次' in content:
                try:
                    parts = content.split('，')
                    for part in parts:
                        if '检测' in part:
                            num_str = ''.join([c for c in part if c.isdigit()])
                            if num_str:
                                total_detections += int(num_str)
                        if '异常' in part:
                            num_str = ''.join([c for c in part if c.isdigit()])
                            if num_str:
                                total_anomalies += int(num_str)
                except:
                    pass
        
        if total_detections == 0:
            return None
        
        avg_anomaly_ratio = (total_anomalies / total_detections * 100) if total_detections > 0 else 0
        
        memory_content = f"近{LONG_TERM_MEMORY_DAYS}天平均异常比例:{avg_anomaly_ratio:.1f}%。"
        
        self.db.add_user_memory(
            user_id=user_id,
            memory_type='long',
            memory_content=memory_content,
            expire_hours=None
        )
        
        return memory_content

    def extract_user_profile(self, user_id: str = 'default'):
        long_memories = self.db.get_user_memory(user_id, memory_type='long')
        posture_data = self.db.get_posture_data(user_id, hours=720)
        
        high_freq_anomaly = None
        anomaly_time_slot = None
        
        if posture_data:
            anomaly_types = defaultdict(int)
            time_slots = defaultdict(int)
            
            for p in posture_data:
                if p['anomaly_flag']:
                    anomaly_types['posture_anomaly'] += 1
                if p['hunchback_flag']:
                    anomaly_types['hunchback'] += 1
                
                if p['anomaly_flag'] or p['hunchback_flag']:
                    ts = p['detect_ts']
                    if isinstance(ts, str):
                        try:
                            ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
                        except:
                            try:
                                ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                            except:
                                continue
                    if ts:
                        slot = f"{ts.hour}:00-{ts.hour+1}:00"
                        time_slots[slot] += 1
            
            if anomaly_types:
                high_freq_anomaly = max(anomaly_types.items(), key=lambda x: x[1])[0]
            
            if time_slots:
                anomaly_time_slot = max(time_slots.items(), key=lambda x: x[1])[0]
        
        improve_strategy = None
        if high_freq_anomaly == 'hunchback':
            improve_strategy = "重点关注驼背问题，建议调整显示器高度，使用腰靠"
        elif high_freq_anomaly == 'posture_anomaly':
            improve_strategy = "注意保持正确坐姿，定时休息活动"
        
        self.db.update_user_profile(
            user_id=user_id,
            high_freq_anomaly=high_freq_anomaly,
            anomaly_time_slot=anomaly_time_slot,
            report_prefer='concise',
            improve_strategy=improve_strategy
        )
        
        return self.db.get_user_profile(user_id)

    def get_profile_summary(self, user_id: str = 'default'):
        profile = self.db.get_user_profile(user_id)
        memories = self.db.get_user_memory(user_id)
        
        return {
            'profile': profile,
            'recent_memories': memories[:5] if memories else []
        }


profile_manager = LightProfileManager()
