from flask import Blueprint, request, jsonify, current_app

rag_bp = Blueprint('rag', __name__)

def get_rag_service():
    return current_app.extensions['hunchback'].get('rag_service')

@rag_bp.route('/add_knowledge', methods=['POST'])
def add_knowledge():
    data = request.json
    content = data.get('content')
    category = data.get('category', 'general')
    source = data.get('source', '')
    
    if not content:
        return jsonify({"error": "Content required"}), 400
    
    rag_service = get_rag_service()
    if not rag_service:
        return jsonify({"error": "RAG service not available"}), 500
    
    try:
        rag_service.add_knowledge(content, category, source)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@rag_bp.route('/add_web_document', methods=['POST'])
def add_web_document():
    data = request.json
    url = data.get('url')
    category = data.get('category', 'web')
    
    if not url:
        return jsonify({"error": "URL required"}), 400
    
    rag_service = get_rag_service()
    if not rag_service:
        return jsonify({"error": "RAG service not available"}), 500
    
    try:
        rag_service.add_web_document(url, category)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@rag_bp.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    top_k = data.get('top_k', 10)
    use_vector = data.get('use_vector', True)
    use_bm25 = data.get('use_bm25', True)
    use_rrf = data.get('use_rrf', True)
    use_cross_encoder = data.get('use_cross_encoder', True)
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    rag_service = get_rag_service()
    if not rag_service:
        return jsonify({"error": "RAG service not available"}), 500
    
    try:
        results = rag_service.search_knowledge(
            query=query,
            top_k=top_k,
            use_vector=use_vector,
            use_bm25=use_bm25,
            use_rrf=use_rrf,
            use_cross_encoder=use_cross_encoder
        )
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@rag_bp.route('/weekly_report', methods=['GET'])
def weekly_report():
    try:
        days = int(request.args.get('days', 7))
        
        rag_service = get_rag_service()
        if not rag_service:
            return jsonify({"error": "RAG service not available"}), 500
        
        report = rag_service.generate_weekly_report(days=days)
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
