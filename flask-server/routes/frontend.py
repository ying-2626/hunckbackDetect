from flask import Blueprint, send_from_directory
import os

frontend_bp = Blueprint('frontend', __name__)

# Use relative paths from this file to the frontend directory
# d:\my-git\hunchback\routes\frontend.py -> d:\my-git\hunchback\frontend
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
FRONTEND_HTML_DIR = os.path.join(FRONTEND_DIR, "html")
FRONTEND_CSS_DIR = os.path.join(FRONTEND_DIR, "css")
FRONTEND_IMG_DIR = os.path.join(FRONTEND_DIR, "img")

@frontend_bp.route('/', methods=['GET'])
def index():
    return send_from_directory(FRONTEND_HTML_DIR, 'index.html')

@frontend_bp.route('/daily_report.html', methods=['GET'])
def daily_report_page():
    return send_from_directory(FRONTEND_HTML_DIR, 'daily_report.html')

@frontend_bp.route('/health_guide.html', methods=['GET'])
def health_guide_page():
    return send_from_directory(FRONTEND_HTML_DIR, 'health_guide.html')

@frontend_bp.route('/login.html', methods=['GET'])
def login_page():
    return send_from_directory(FRONTEND_HTML_DIR, 'login.html')

@frontend_bp.route('/css/<path:filename>', methods=['GET'])
def frontend_css(filename):
    return send_from_directory(FRONTEND_CSS_DIR, filename)

@frontend_bp.route('/img/<path:filename>', methods=['GET'])
def frontend_img(filename):
    return send_from_directory(FRONTEND_IMG_DIR, filename)

@frontend_bp.route('/favicon.ico', methods=['GET'])
def favicon():
    return ('', 204)
