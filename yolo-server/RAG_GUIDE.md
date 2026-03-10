# 高级 RAG 系统使用指南

## 系统架构

本项目已升级为高级 RAG 系统，包含以下特性：

1. **Milvus 向量数据库** - 使用 HNSW 索引加速向量检索
2. **BM25 关键词检索** - 基于词频-逆文档频率的传统检索
3. **倒数秩融合 (RRF)** - 融合两路检索结果
4. **文本去重** - 基于内容相似度去重
5. **Cross-Encoder 精排** - 使用交叉编码器重新排序
6. **文档处理** - 网页抓取 + 10% overlap 文本分块

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 部署 Milvus

使用 Docker Compose 启动 Milvus：

```bash
docker-compose up -d
```

服务说明：
- Milvus: `localhost:19530`
- Attu (Milvus 管理界面): `http://localhost:8000`

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并配置：

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的配置
```

配置项说明：
- `DASHSCOPE_API_KEY`: 阿里云 DashScope API Key
- `MILVUS_HOST`: Milvus 服务器地址 (默认 localhost)
- `MILVUS_PORT`: Milvus 端口 (默认 19530)
- `BM25_INDEX_PATH`: BM25 索引保存路径

### 4. 使用高级 RAG 服务

#### 基础使用

```python
from rag_advanced import AdvancedRAGService

# 初始化 RAG 服务
rag = AdvancedRAGService(
    use_milvus=True,
    use_bm25=True,
    use_fusion=True
)

# 添加知识
rag.add_knowledge(
    content="长时间低头会导致颈椎前倾，建议调整显示器高度至视线平齐。",
    category="ergonomics"
)

# 搜索知识
results = rag.search_knowledge(
    query="如何改善不良坐姿？",
    top_k=5
)

# 生成周报
report = rag.generate_weekly_report(days=7)
print(report)
```

#### 添加网络文档

```python
# 直接从 URL 添加文档
rag.add_web_document(
    url="https://example.com/health-posture",
    category="health"
)
```

#### 自定义检索策略

```python
# 只使用向量检索
results = rag.search_knowledge(
    query="颈椎问题",
    use_vector=True,
    use_bm25=False
)

# 只使用 BM25 检索
results = rag.search_knowledge(
    query="腰椎",
    use_vector=False,
    use_bm25=True
)

# 禁用 Cross-Encoder 精排（更快）
results = rag.search_knowledge(
    query="坐姿",
    use_cross_encoder=False
)
```

## 模块说明

### 1. milvus_client.py - Milvus 客户端

```python
from milvus_client import MilvusClient

# 创建客户端
client = MilvusClient(
    host="localhost",
    port="19530",
    collection_name="knowledge_base",
    dim=1536
)

# 插入数据
client.insert(
    contents=["文档内容1", "文档内容2"],
    categories=["general", "health"],
    sources=["source1", "source2"],
    chunk_indices=[0, 1],
    embeddings=[[...], [...]]
)

# 搜索
results = client.search(query_embedding=[...], top_k=10)
```

### 2. document_processor.py - 文档处理

```python
from document_processor import DocumentProcessor

processor = DocumentProcessor(
    chunk_size=512,
    overlap_percent=0.1  # 10% overlap
)

# 处理 URL
chunks = processor.process_url("https://example.com/article")

# 处理纯文本
chunks = processor.process_text(
    text="长文本内容...",
    title="文章标题",
    source="来源"
)

for chunk in chunks:
    print(f"Chunk {chunk['chunk_index']}: {chunk['content'][:100]}...")
```

### 3. bm25_retriever.py - BM25 检索

```python
from bm25_retriever import BM25Retriever

retriever = BM25Retriever(index_path="./data/bm25.pkl")

# 添加文档
retriever.add_documents([
    {"content": "文档1内容", "category": "a"},
    {"content": "文档2内容", "category": "b"}
])

# 搜索
results = retriever.search("查询关键词", top_k=10)

# 保存/加载索引
retriever.save_index()
retriever.load_index()
```

### 4. retrieval_fusion.py - 检索融合

```python
from retrieval_fusion import RetrievalFusion

fusion = RetrievalFusion(
    rrf_k=60,
    cross_encoder_model="BAAI/bge-reranker-base"
)

# RRF 融合
fused = fusion.rrf_fusion([vector_results, bm25_results])

# 去重
deduped = fusion.deduplicate(fused, similarity_threshold=0.8)

# Cross-Encoder 精排
reranked = fusion.cross_encoder_rerank(
    query="查询",
    results=deduped,
    top_k=10
)

# 完整流水线
final_results = fusion.full_retrieval_pipeline(
    query="查询",
    vector_results=vector_results,
    bm25_results=bm25_results,
    rerank_top_k=10
)
```

## 与 Flask 应用集成

更新 `app.py` 使用新的 RAG 服务：

```python
from flask import Flask, request, jsonify
from inference import YOLOService, RequestHandler
from rag_advanced import AdvancedRAGService

app = Flask(__name__)

model_path = r"d:\my-git\hunchback\SittingWatch\SittingWatch-main\YOLOv8\runs\detect\train3\weights\best.pt"
yolo_service = YOLOService(model_path)
request_handler = RequestHandler(yolo_service)

# 使用高级 RAG 服务
rag_service = AdvancedRAGService(
    use_milvus=True,
    use_bm25=True,
    use_fusion=True
)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    if not image_file.filename.endswith('.jpg'):
        return jsonify({"error": "Only JPG images are supported"}), 400
    
    return request_handler.handle_request(image_file)

@app.route('/report', methods=['GET'])
def get_report():
    try:
        days = int(request.args.get('days', 7))
        report = rag_service.generate_weekly_report(days=days)
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_knowledge', methods=['POST'])
def add_knowledge():
    data = request.json
    content = data.get('content')
    category = data.get('category', 'general')
    source = data.get('source', '')
    if not content:
        return jsonify({"error": "Content required"}), 400
    
    rag_service.add_knowledge(content, category, source)
    return jsonify({"status": "success"})

@app.route('/add_web_document', methods=['POST'])
def add_web_document():
    data = request.json
    url = data.get('url')
    category = data.get('category', 'web')
    if not url:
        return jsonify({"error": "URL required"}), 400
    
    try:
        rag_service.add_web_document(url, category)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    top_k = data.get('top_k', 10)
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    results = rag_service.search_knowledge(query, top_k=top_k)
    return jsonify({"results": results})

if __name__ == '__main__':
    pass
```

## API 接口

### 1. 添加知识

```bash
curl -X POST http://localhost:5000/add_knowledge \
  -H "Content-Type: application/json" \
  -d '{
    "content": "每坐1小时应站立活动5分钟",
    "category": "health_tips",
    "source": "manual"
  }'
```

### 2. 添加网络文档

```bash
curl -X POST http://localhost:5000/add_web_document \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/health-article",
    "category": "health"
  }'
```

### 3. 搜索

```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何改善坐姿",
    "top_k": 5
  }'
```

### 4. 获取周报

```bash
curl "http://localhost:5000/report?days=7"
```

## 故障排除

### Milvus 连接失败

1. 检查 Docker 容器状态：
   ```bash
   docker-compose ps
   ```

2. 查看日志：
   ```bash
   docker-compose logs standalone
   ```

3. 确认端口 19530 未被占用

### 依赖安装问题

如果安装 sentence-transformers 遇到问题，可以使用 CPU 版本：

```bash
pip install sentence-transformers --no-deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 降级使用

如果不想使用 Milvus，可以初始化时禁用：

```python
rag = AdvancedRAGService(use_milvus=False)
```

这样会回退到 SQLite 存储，保持现有功能。

## 性能优化建议

1. **首次使用**：Cross-Encoder 模型会在首次使用时下载，需要网络连接
2. **批量插入**：使用 `add_knowledge` 时，系统会自动分块处理
3. **BM25 索引**：定期使用 `save_bm25_index()` 保存索引，避免重复构建
4. **HNSW 参数**：可在 `milvus_client.py` 中调整 M 和 efConstruction 参数

## MCP 集成（可选）

如需集成 MCP 网络搜索工具，可以在 `rag_advanced.py` 中添加：

```python
# 在 AdvancedRAGService 类中添加
def search_web_and_add(self, query: str, category: str = "web_search"):
    # 使用 MCP 工具搜索网络
    # 获取搜索结果的网页链接
    # 调用 add_web_document() 添加每个链接
    pass
```

## 总结

本 RAG 系统提供了从简单到高级的完整检索方案：

- ✅ Milvus + HNSW 向量检索
- ✅ BM25 关键词匹配
- ✅ RRF 结果融合
- ✅ 内容去重
- ✅ Cross-Encoder 精排
- ✅ 10% overlap 文本分块
- ✅ 网络文档抓取
- ✅ 向后兼容（可降级使用）
