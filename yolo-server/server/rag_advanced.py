import os
import datetime
import json
import numpy as np
from typing import List, Dict, Any, Optional
from db import db

try:
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

try:
    from milvus_client import MilvusClient
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    from bm25_retriever import BM25Retriever
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from retrieval_fusion import RetrievalFusion
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False

try:
    from document_processor import DocumentProcessor
    DOC_PROCESSOR_AVAILABLE = True
except ImportError:
    DOC_PROCESSOR_AVAILABLE = False

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

class EmbeddingService:
    def get_embedding(self, text: str) -&gt; List[float]:
        raise NotImplementedError

class MockEmbeddingService(EmbeddingService):
    def __init__(self, dim=1536):
        self.dim = dim

    def get_embedding(self, text: str) -&gt; List[float]:
        vec = np.random.rand(self.dim).astype(np.float32)
        return (vec / np.linalg.norm(vec)).tolist()

class DashScopeEmbeddingService(EmbeddingService):
    def get_embedding(self, text: str) -&gt; List[float]:
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("DashScope not installed")
        resp = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v1,
            input=text
        )
        if resp.status_code == 200:
            return resp.output['embeddings'][0]['embedding']
        else:
            raise Exception(f"DashScope Error: {resp}")

class LLMService:
    def generate(self, prompt: str) -&gt; str:
        raise NotImplementedError

class MockLLMService(LLMService):
    def generate(self, prompt: str) -&gt; str:
        return f"【Mock LLM Response】\n收到 Prompt: {prompt[:50]}...\n(请配置 DashScope API Key 以使用真实 LLM)"

class DashScopeLLMService(LLMService):
    def generate(self, prompt: str) -&gt; str:
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

class AdvancedRAGService:
    def __init__(
        self,
        embedding_service: EmbeddingService = None,
        llm_service: LLMService = None,
        use_milvus: bool = True,
        use_bm25: bool = True,
        use_fusion: bool = True,
        milvus_host: str = None,
        milvus_port: str = None,
        bm25_index_path: str = None
    ):
        if embedding_service:
            self.embedding_service = embedding_service
        else:
            if (DASHSCOPE_API_KEY or os.environ.get("DASHSCOPE_API_KEY")) and DASHSCOPE_AVAILABLE:
                self.embedding_service = DashScopeEmbeddingService()
            else:
                self.embedding_service = MockEmbeddingService()
        
        if llm_service:
            self.llm_service = llm_service
        else:
            if (DASHSCOPE_API_KEY or os.environ.get("DASHSCOPE_API_KEY")) and DASHSCOPE_AVAILABLE:
                self.llm_service = DashScopeLLMService()
            else:
                self.llm_service = MockLLMService()
        
        self.use_milvus = use_milvus and MILVUS_AVAILABLE
        self.use_bm25 = use_bm25 and BM25_AVAILABLE
        self.use_fusion = use_fusion and FUSION_AVAILABLE
        
        self.milvus_client = None
        if self.use_milvus:
            try:
                self.milvus_client = MilvusClient(
                    host=milvus_host,
                    port=milvus_port
                )
            except Exception as e:
                print(f"Failed to initialize Milvus: {e}. Falling back to SQLite.")
                self.use_milvus = False
        
        self.bm25_retriever = None
        if self.use_bm25:
            self.bm25_retriever = BM25Retriever(index_path=bm25_index_path)
            if bm25_index_path:
                self.bm25_retriever.load_index()
        
        self.retrieval_fusion = None
        if self.use_fusion:
            self.retrieval_fusion = RetrievalFusion()
        
        self.doc_processor = None
        if DOC_PROCESSOR_AVAILABLE:
            self.doc_processor = DocumentProcessor(overlap_percent=0.1)

    def add_knowledge(
        self,
        content: str,
        category: str = "general",
        source: str = ""
    ):
        if self.doc_processor:
            chunks = self.doc_processor.process_text(content, source=source)
        else:
            chunks = [{
                'content': content,
                'source': source,
                'chunk_index': 0
            }]
        
        contents = [chunk['content'] for chunk in chunks]
        sources = [chunk.get('source', '') for chunk in chunks]
        chunk_indices = [chunk.get('chunk_index', 0) for chunk in chunks]
        categories = [category] * len(chunks)
        embeddings = [self.embedding_service.get_embedding(c) for c in contents]
        
        if self.use_milvus and self.milvus_client:
            self.milvus_client.insert(
                contents=contents,
                categories=categories,
                sources=sources,
                chunk_indices=chunk_indices,
                embeddings=embeddings
            )
        else:
            for c, cat, emb in zip(contents, categories, embeddings):
                db.add_knowledge(c, cat, emb)
        
        if self.use_bm25 and self.bm25_retriever:
            bm25_docs = []
            for i, chunk in enumerate(chunks):
                bm25_docs.append({
                    'content': chunk['content'],
                    'category': category,
                    'source': chunk.get('source', ''),
                    'chunk_index': chunk.get('chunk_index', 0)
                })
            self.bm25_retriever.add_documents(bm25_docs)

    def add_web_document(self, url: str, category: str = "web"):
        if not self.doc_processor:
            raise RuntimeError("Document processor not available")
        
        chunks = self.doc_processor.process_url(url)
        
        contents = [chunk['content'] for chunk in chunks]
        sources = [chunk.get('source', url) for chunk in chunks]
        chunk_indices = [chunk.get('chunk_index', 0) for chunk in chunks]
        categories = [category] * len(chunks)
        embeddings = [self.embedding_service.get_embedding(c) for c in contents]
        
        if self.use_milvus and self.milvus_client:
            self.milvus_client.insert(
                contents=contents,
                categories=categories,
                sources=sources,
                chunk_indices=chunk_indices,
                embeddings=embeddings
            )
        else:
            for c, cat, emb in zip(contents, categories, embeddings):
                db.add_knowledge(c, cat, emb)
        
        if self.use_bm25 and self.bm25_retriever:
            bm25_docs = []
            for i, chunk in enumerate(chunks):
                bm25_docs.append({
                    'content': chunk['content'],
                    'category': category,
                    'source': chunk.get('source', url),
                    'chunk_index': chunk.get('chunk_index', 0)
                })
            self.bm25_retriever.add_documents(bm25_docs)

    def search_knowledge(
        self,
        query: str,
        top_k: int = 10,
        use_vector: bool = True,
        use_bm25: bool = True,
        use_rrf: bool = True,
        use_dedup: bool = True,
        use_cross_encoder: bool = True
    ) -&gt; List[Dict[str, Any]]:
        vector_results = []
        bm25_results = []
        
        if use_vector:
            query_embedding = self.embedding_service.get_embedding(query)
            if self.use_milvus and self.milvus_client:
                vector_results = self.milvus_client.search(query_embedding, top_k=top_k)
            else:
                vector_results = db.search_knowledge(query_embedding, top_k=top_k)
                for r in vector_results:
                    r['retriever'] = 'vector'
        
        if use_bm25 and self.use_bm25 and self.bm25_retriever:
            bm25_results = self.bm25_retriever.search(query, top_k=top_k)
        
        if use_rrf and self.use_fusion and self.retrieval_fusion and vector_results and bm25_results:
            return self.retrieval_fusion.full_retrieval_pipeline(
                query=query,
                vector_results=vector_results,
                bm25_results=bm25_results,
                rerank_top_k=top_k,
                use_cross_encoder=use_cross_encoder
            )
        elif vector_results:
            return vector_results[:top_k]
        elif bm25_results:
            return bm25_results[:top_k]
        else:
            return []

    def generate_weekly_report(self, user_id: str = "default", days: int = 7) -&gt; str:
        now = datetime.datetime.now()
        start_time = now - datetime.timedelta(days=days)
        
        records = db.get_records(limit=2000, start_time=start_time)
        
        recent_records = []
        for r in records:
            try:
                if isinstance(r['timestamp'], str):
                    try:
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

        total_count = len(recent_records)
        bad_count = sum(1 for r in recent_records if r['class_name'] == 'sitting_bad')
        bad_ratio = (bad_count / total_count) * 100 if total_count &gt; 0 else 0
        
        summary_text = f"本周共检测 {total_count} 次，不良坐姿 {bad_count} 次，占比 {bad_ratio:.1f}%。"
        
        query_text = f"不良坐姿占比 {bad_ratio:.1f}%，如何改善？"
        knowledge_items = self.search_knowledge(query_text, top_k=5)
        
        knowledge_context = "\n".join([f"- {item['content']}" for item in knowledge_items])
        if not knowledge_items:
            knowledge_context = "暂无相关知识库建议。"
        
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

    def save_bm25_index(self, path: str = None):
        if self.use_bm25 and self.bm25_retriever:
            self.bm25_retriever.save_index(path)

if __name__ == "__main__":
    rag = AdvancedRAGService(use_milvus=False)
    rag.add_knowledge("长时间低头会导致颈椎前倾，建议调整显示器高度至视线平齐。", "ergonomics")
    rag.add_knowledge("每坐1小时应站立活动5分钟，缓解腰椎压力。", "health_tips")
    
    print("生成测试报告...")
    print(rag.generate_weekly_report())
