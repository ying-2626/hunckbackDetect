import os
import numpy as np
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusException
)

class MilvusClient:
    def __init__(
        self,
        host: str = None,
        port: str = None,
        collection_name: str = "knowledge_base",
        dim: int = 1536
    ):
        self.host = host or os.environ.get("MILVUS_HOST", "localhost")
        self.port = port or os.environ.get("MILVUS_PORT", "19530")
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None
        self._connect()
        self._init_collection()

    def _connect(self):
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"Connected to Milvus at {self.host}:{self.port}")
        except MilvusException as e:
            print(f"Failed to connect to Milvus: {e}")
            raise

    def _init_collection(self):
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print(f"Loaded existing collection: {self.collection_name}")
        else:
            self._create_collection()

    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        
        schema = CollectionSchema(fields, description="Knowledge base for RAG")
        self.collection = Collection(self.collection_name, schema)
        
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 200
            }
        }
        
        self.collection.create_index(field_name="embedding", index_params=index_params)
        self.collection.load()
        print(f"Created and loaded collection: {self.collection_name} with HNSW index")

    def insert(
        self,
        contents: List[str],
        categories: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        chunk_indices: Optional[List[int]] = None,
        embeddings: List[List[float]] = None
    ) -> List[int]:
        if categories is None:
            categories = ["general"] * len(contents)
        if sources is None:
            sources = [""] * len(contents)
        if chunk_indices is None:
            chunk_indices = [0] * len(contents)
        
        data = [
            contents,
            categories,
            sources,
            chunk_indices,
            embeddings
        ]
        
        result = self.collection.insert(data)
        self.collection.flush()
        return result.primary_keys

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 128}
        }
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["content", "category", "source", "chunk_index"]
        )
        
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "content": hit.entity.get("content"),
                    "category": hit.entity.get("category"),
                    "source": hit.entity.get("source"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "score": hit.score
                })
        
        return formatted_results

    def get_all(self, limit: int = 1000) -> List[Dict[str, Any]]:
        expr = "id >= 0"
        results = self.collection.query(
            expr=expr,
            output_fields=["id", "content", "category", "source", "chunk_index"],
            limit=limit
        )
        return results

    def delete(self, ids: List[int]):
        expr = f"id in {ids}"
        self.collection.delete(expr)
        self.collection.flush()

    def drop_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"Dropped collection: {self.collection_name}")

    def close(self):
        connections.disconnect("default")
