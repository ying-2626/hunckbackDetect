import os
import logging
from flask import Flask
from flask_mail import Mail
from flask_cors import CORS

from core.analyzer import Analyzer
from core.capture import CameraCapture
from core.scheduler import Scheduler
from core.classifier import Classifier
from utils.config import SERVER_CONFIG

from routes.auth import auth_bp
from routes.frontend import frontend_bp
from routes.report import report_bp
from routes.analysis import analysis_bp
from routes.rag import rag_bp

try:
    from RAG import AdvancedRAGService
    RAG_AVAILABLE = True
except ImportError:
    try:
        from .RAG import AdvancedRAGService
        RAG_AVAILABLE = True
    except ImportError:
        RAG_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
CORS(app)
mail = Mail(app)

# Check if running on Vercel
IS_VERCEL = os.environ.get('VERCEL') == '1'

# Global instances
camera_instance = None
s = None
analyzer_instance = None
classifier_instance = None
rag_service_instance = None

def init_components():
    global camera_instance, s, analyzer_instance, classifier_instance, rag_service_instance
    
    if not IS_VERCEL:
        camera_instance = CameraCapture()  # Global unique camera instance
        # Scheduler takes mail, app, and camera_instance
        s = Scheduler(mail, app, camera_instance)
    else:
        camera_instance = None
        s = None

    analyzer_instance = Analyzer()  # Global Analyzer instance
    classifier_instance = Classifier()  # Global Classifier instance
    
    rag_service_instance = None
    if RAG_AVAILABLE:
        try:
            rag_service_instance = AdvancedRAGService(
                use_milvus=False,
                use_bm25=True,
                use_fusion=True
            )
            print("RAG service initialized successfully")
        except Exception as e:
            print(f"Failed to initialize RAG service: {e}")

    # Store in app extensions for Blueprints to access
    app.extensions['hunchback'] = {
        'camera_instance': camera_instance,
        'analyzer_instance': analyzer_instance,
        'classifier_instance': classifier_instance,
        'rag_service': rag_service_instance,
        'is_vercel': IS_VERCEL
    }

# Initialize components
init_components()

# Register Blueprints
app.register_blueprint(frontend_bp)
app.register_blueprint(auth_bp, url_prefix='/api')
app.register_blueprint(report_bp, url_prefix='/api')
app.register_blueprint(analysis_bp)
app.register_blueprint(rag_bp, url_prefix='/api/rag')

def start_background_tasks():
    if not IS_VERCEL and s:
        s.start()

if __name__ == '__main__':
    try:
        start_background_tasks()
        app.run(host='0.0.0.0', port=SERVER_CONFIG['PORT'])
    finally:
        if camera_instance:
            camera_instance.release()
