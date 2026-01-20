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

class LLMService:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class MockLLMService(LLMService):
    def generate(self, prompt: str) -> str:
        return f"【Mock LLM Response】\n收到 Prompt: {prompt[:50]}...\n(请配置 DashScope API Key 以使用真实 LLM)"

class DashScopeLLMService(LLMService):
    def generate(self, prompt: str) -> str:
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("DashScope not installed")
        
        try:
            resp = dashscope.Generation.call(
                model=dashscope.Generation.Models.qwen_turbo,
                prompt=prompt
            )
            if resp.status_code == 200:
                return resp.output.text
            else:
                return f"Error: {resp.code} - {resp.message}"
        except Exception as e:
            return f"Exception: {str(e)}"

class RAGService:
    def __init__(self, embedding_service: EmbeddingService = None, llm_service: LLMService = None):
        # 初始化 Embedding Service
        if embedding_service:
            self.embedding_service = embedding_service
        else:
            if os.environ.get("DASHSCOPE_API_KEY") and DASHSCOPE_AVAILABLE:
                self.embedding_service = DashScopeEmbeddingService()
            else:
                self.embedding_service = MockEmbeddingService()
        
        # 初始化 LLM Service
        if llm_service:
            self.llm_service = llm_service
        else:
            if os.environ.get("DASHSCOPE_API_KEY") and DASHSCOPE_AVAILABLE:
                self.llm_service = DashScopeLLMService()
            else:
                self.llm_service = MockLLMService()

    def add_knowledge(self, content: str, category: str = "general"):
        """添加知识到知识库"""
        embedding = self.embedding_service.get_embedding(content)
        db.add_knowledge(content, category, embedding)

    def generate_weekly_report(self, user_id: str = "default", days: int = 7) -> str:
        """生成周报"""
        # 1. 获取最近检测记录
        now = datetime.datetime.now()
        start_time = now - datetime.timedelta(days=days)
        
        # 使用 DB 层的过滤
        records = db.get_records(limit=2000, start_time=start_time)
        
        recent_records = []
        for r in records:
            # 这里的过滤其实已经是多余的了，因为 DB 已经过滤了
            # 但为了兼容 timestamp 格式解析（如果 DB 中存储的是字符串），我们还是保留解析逻辑，但不再需要过滤
            try:
                if isinstance(r['timestamp'], str):
                    try:
                        # 尝试解析，确保数据有效
                        ts = datetime.datetime.strptime(r['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
                    except:
                        ts = datetime.datetime.strptime(r['timestamp'], "%Y-%m-%d %H:%M:%S")
                elif isinstance(r['timestamp'], datetime.datetime):
                    ts = r['timestamp']
                else:
                    continue
                recent_records.append(r)
            except:
                continue
        
        if not recent_records:
            return "本周暂无检测记录。"

        # 2. 统计数据
        total_count = len(recent_records)
        bad_count = sum(1 for r in recent_records if r['class_name'] == 'sitting_bad')
        bad_ratio = (bad_count / total_count) * 100 if total_count > 0 else 0
        
        summary_text = f"本周共检测 {total_count} 次，不良坐姿 {bad_count} 次，占比 {bad_ratio:.1f}%。"
        
        # 3. 检索知识库 (RAG)
        query_text = f"不良坐姿占比 {bad_ratio:.1f}%，如何改善？"
        query_embedding = self.embedding_service.get_embedding(query_text)
        knowledge_items = db.search_knowledge(query_embedding, top_k=3)
        
        knowledge_context = "\n".join([f"- {item['content']}" for item in knowledge_items])
        if not knowledge_items:
            knowledge_context = "暂无相关知识库建议。"
        
        # 4. 生成报告 (LLM)
        prompt = f"""
你是一个专业的坐姿健康助手。请根据以下用户数据和参考知识，生成一份温馨、专业的周报。

【用户数据】
时间范围: {start_time.strftime('%Y-%m-%d')} 至 {now.strftime('%Y-%m-%d')}
{summary_text}

【参考知识】
{knowledge_context}

【要求】
1. 分析用户的坐姿情况。
2. 结合参考知识给出具体的改进建议。
3. 语气亲切，鼓励用户保持健康。
4. 格式清晰，使用 Markdown。
"""
        return self.llm_service.generate(prompt)

# 简单测试入口
if __name__ == "__main__":
    rag = RAGService()
    # 模拟添加一些知识
    rag.add_knowledge("长时间低头会导致颈椎前倾，建议调整显示器高度至视线平齐。", "ergonomics")
    rag.add_knowledge("每坐1小时应站立活动5分钟，缓解腰椎压力。", "health_tips")
    
    print("生成测试报告...")
    print(rag.generate_weekly_report())
