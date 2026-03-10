import os
import jieba
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import pickle

class BM25Retriever:
    def __init__(self, index_path: str = None):
        self.index_path = index_path
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        words = jieba.lcut(text)
        return [word for word in words if word.strip()]

    def add_documents(self, documents: List[Dict[str, Any]]):
        for doc in documents:
            self.documents.append(doc)
            tokens = self._tokenize(doc['content'])
            self.tokenized_corpus.append(tokens)
        
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        doc_score_pairs = list(zip(self.documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, score in doc_score_pairs[:top_k]:
            result = doc.copy()
            result['score'] = float(score)
            result['retriever'] = 'bm25'
            results.append(result)
        
        return results

    def save_index(self, path: str = None):
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No index path provided")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data = {
            'documents': self.documents,
            'tokenized_corpus': self.tokenized_corpus
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def load_index(self, path: str = None):
        load_path = path or self.index_path
        if not load_path or not os.path.exists(load_path):
            return False
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.tokenized_corpus = data['tokenized_corpus']
        
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        return True
