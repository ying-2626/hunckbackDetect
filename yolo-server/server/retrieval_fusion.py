import hashlib
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

class RetrievalFusion:
    def __init__(
        self,
        rrf_k: int = 60,
        cross_encoder_model: str = "BAAI/bge-reranker-base"
    ):
        self.rrf_k = rrf_k
        self.cross_encoder = None
        self.cross_encoder_model = cross_encoder_model
        
        if CROSS_ENCODER_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model)
            except Exception as e:
                print(f"Failed to load CrossEncoder: {e}")

    def rrf_fusion(
        self,
        results_list: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        doc_scores = {}
        
        for results in results_list:
            for rank, doc in enumerate(results, start=1):
                doc_id = self._get_doc_id(doc)
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'score': 0.0
                    }
                doc_scores[doc_id]['score'] += 1.0 / (self.rrf_k + rank)
        
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        fused_results = []
        for item in sorted_docs:
            result = item['doc'].copy()
            result['rrf_score'] = item['score']
            fused_results.append(result)
        
        return fused_results

    def _get_doc_id(self, doc: Dict[str, Any]) -> str:
        if 'id' in doc:
            return str(doc['id'])
        content = doc.get('content', '')
        return hashlib.md5(content.encode()).hexdigest()

    def _content_similarity(self, text1: str, text2: str) -> float:
        return SequenceMatcher(None, text1, text2).ratio()

    def deduplicate(
        self,
        results: List[Dict[str, Any]],
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        unique_results = []
        seen_contents = []
        
        for result in results:
            content = result.get('content', '')
            is_duplicate = False
            
            for seen_content in seen_contents:
                if self._content_similarity(content, seen_content) &gt;= similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_contents.append(content)
        
        return unique_results

    def cross_encoder_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10
    ) -&gt; List[Dict[str, Any]]:
        if not self.cross_encoder or not results:
            return results[:top_k]
        
        pairs = [[query, doc['content']] for doc in results]
        scores = self.cross_encoder.predict(pairs)
        
        for i, doc in enumerate(results):
            doc['cross_encoder_score'] = float(scores[i])
        
        reranked = sorted(
            results,
            key=lambda x: x['cross_encoder_score'],
            reverse=True
        )
        
        return reranked[:top_k]

    def full_retrieval_pipeline(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        rrf_k: Optional[int] = None,
        dedup_threshold: float = 0.8,
        rerank_top_k: int = 10,
        use_cross_encoder: bool = True
    ) -&gt; List[Dict[str, Any]]:
        k = rrf_k or self.rrf_k
        
        for i, doc in enumerate(vector_results):
            doc['retriever'] = 'vector'
        
        fused = self.rrf_fusion([vector_results, bm25_results])
        
        deduped = self.deduplicate(fused, dedup_threshold)
        
        if use_cross_encoder and self.cross_encoder:
            final_results = self.cross_encoder_rerank(query, deduped, rerank_top_k)
        else:
            final_results = deduped[:rerank_top_k]
        
        return final_results
