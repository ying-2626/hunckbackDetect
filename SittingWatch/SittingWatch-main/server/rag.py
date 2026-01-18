import os
import datetime
import json
import random
import numpy as np
from typing import List, Dict, Any
from db import db

# 尝试导入 dashscope，如果不存在则使用 Mock
try:
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

class EmbeddingService:
    def get_embedding(self, text: str) -> List[float]:
        raise NotImplementedError

class MockEmbeddingService(EmbeddingService):
    def __init__(self, dim=1536):
        self.dim = dim

    def get_embedding(self, text: str) -> List[float]:
        # 返回随机向量用于测试
        # 实际使用中请替换为真实模型
        vec = np.random.rand(self.dim).astype(np.float32)
        return (vec / np.linalg.norm(vec)).tolist()

class DashScopeEmbeddingService(EmbeddingService):
    def get_embedding(self, text: str) -> List[float]:
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("DashScope not installed")
        # 需要设置 DASHSCOPE_API_KEY 环境变量
        resp = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v1,
            input=text
        )
        if resp.status_code == 200:
            return resp.output['embeddings'][0]['embedding']
        else:
            raise Exception(f"DashScope Error: {resp}")

class RAGService:
    def __init__(self, embedding_service: EmbeddingService = None):
        if embedding_service:
            self.embedding_service = embedding_service
        else:
            # 默认使用 Mock，除非配置了 Key
            if os.environ.get("DASHSCOPE_API_KEY") and DASHSCOPE_AVAILABLE:
                self.embedding_service = DashScopeEmbeddingService()
            else:
                self.embedding_service = MockEmbeddingService()

    def add_knowledge(self, content: str, category: str = "general"):
        """添加知识到知识库"""
        embedding = self.embedding_service.get_embedding(content)
        db.add_knowledge(content, category, embedding)

    def generate_weekly_report(self, user_id: str = "default", days: int = 7) -> str:
        """生成周报"""
        # 1. 获取最近检测记录
        records = db.get_records(limit=1000) # 获取足够多的记录
        
        # 简单过滤最近N天 (实际生产中应在 SQL 中过滤)
        now = datetime.datetime.now()
        start_time = now - datetime.timedelta(days=days)
        
        recent_records = []
        for r in records:
            # timestamp 格式可能是字符串或 datetime对象，视 sqlite adapter 而定
            # 这里简化处理，假设是字符串
            try:
                ts = datetime.datetime.strptime(r['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
            except:
                try:
                    ts = datetime.datetime.strptime(r['timestamp'], "%Y-%m-%d %H:%M:%S")
                except:
                    continue
            
            if ts > start_time:
                recent_records.append(r)
        
        if not recent_records:
            return "本周暂无检测记录。"

        # 2. 统计数据
        total_count = len(recent_records)
        bad_count = sum(1 for r in recent_records if r['class_name'] == 'sitting_bad')
        bad_ratio = (bad_count / total_count) * 100 if total_count > 0 else 0
        
        summary_text = f"本周共检测 {total_count} 次，不良坐姿 {bad_count} 次，占比 {bad_ratio:.1f}%。"
        
        # 3. 检索知识库 (RAG)
        # 构造查询向量
        query_embedding = self.embedding_service.get_embedding(f"不良坐姿占比 {bad_ratio:.1f}%，如何改善？")
        knowledge_items = db.search_knowledge(query_embedding, top_k=3)
        
        knowledge_context = "\n".join([f"- {item['content']}" for item in knowledge_items])
        
        # 4. 生成报告 (Mock LLM)
        # 实际项目中这里应调用 LLM API
        report = f"""
【坐姿健康周报】
时间范围: {start_time.strftime('%Y-%m-%d')} 至 {now.strftime('%Y-%m-%d')}

【数据概览】
{summary_text}

【健康建议】 (基于 RAG 检索)
{knowledge_context if knowledge_items else "暂无相关知识库建议。"}

【改进计划】
建议您每工作45分钟休息一次，并尝试上述建议中的拉伸动作。
"""
        return report

# 简单测试入口
if __name__ == "__main__":
    rag = RAGService()
    # 模拟添加一些知识
    rag.add_knowledge("长时间低头会导致颈椎前倾，建议调整显示器高度至视线平齐。", "ergonomics")
    rag.add_knowledge("每坐1小时应站立活动5分钟，缓解腰椎压力。", "health_tips")
    
    print("生成测试报告...")
    print(rag.generate_weekly_report())
