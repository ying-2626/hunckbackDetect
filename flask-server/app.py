import os
import logging
from flask import Flask
from flask_mail import Mail
from flask_cors import CORS

from core.analyzer import Analyzer
from core.capture import CameraCapture
from core.scheduler import Scheduler
from core.classifier import Classifier
from utils.config import SERVER_CONFIG, RAG_ENABLED

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


def init_knowledge_base_if_needed(rag_service):
    if not rag_service:
        return
    try:
        from RAG.db import db as rag_db
        results = rag_db.search_knowledge([0.0] * 1536, top_k=1)
        if not results:
            print("知识库为空，正在初始化...")
            try:
                from RAG.init_knowledge import init_knowledge_base
                init_knowledge_base()
                print("知识库初始化完成！")
            except Exception as e:
                print(f"初始化知识库失败: {e}")
        else:
            print("知识库已存在，跳过初始化")
    except Exception as e:
        print(f"检查知识库状态失败: {e}")


app = Flask(__name__)
CORS(app)
mail = Mail(app)

IS_VERCEL = os.environ.get('VERCEL') == '1'

camera_instance = None
s = None
analyzer_instance = None
classifier_instance = None
rag_service_instance = None
db_manager_instance = None
profile_manager_instance = None


def init_components():
    global camera_instance, s, analyzer_instance, classifier_instance, rag_service_instance
    global db_manager_instance, profile_manager_instance
    
    if not IS_VERCEL:
        camera_instance = CameraCapture()
        s = Scheduler(mail, app, camera_instance)
    else:
        camera_instance = None
        s = None

    analyzer_instance = Analyzer()
    classifier_instance = Classifier()
    
    try:
        from storage.local_db_manager import db_manager
        db_manager_instance = db_manager
        print("数据库管理器初始化成功")
    except Exception as e:
        print(f"数据库管理器初始化失败: {e}")
    
    try:
        from profile.light_profile_manager import profile_manager
        profile_manager_instance = profile_manager
        print("用户画像管理器初始化成功")
    except Exception as e:
        print(f"用户画像管理器初始化失败: {e}")
    
    rag_service_instance = None
    if RAG_AVAILABLE and RAG_ENABLED:
        try:
            rag_service_instance = AdvancedRAGService(
                use_milvus=False,
                use_bm25=True,
                use_fusion=True
            )
            print("RAG 服务初始化成功")
            init_knowledge_base_if_needed(rag_service_instance)
        except Exception as e:
            print(f"RAG 服务初始化失败: {e}")

    app.extensions['hunchback'] = {
        'camera_instance': camera_instance,
        'analyzer_instance': analyzer_instance,
        'classifier_instance': classifier_instance,
        'rag_service': rag_service_instance,
        'db_manager': db_manager_instance,
        'profile_manager': profile_manager_instance,
        'is_vercel': IS_VERCEL
    }


init_components()

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
