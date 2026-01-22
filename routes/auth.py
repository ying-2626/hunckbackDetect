from flask import Blueprint, request, jsonify
from services.user_service import register_user, verify_user

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def api_register():
    data = request.get_json()

    print("收到登录请求")  # 添加调试输出
    print("请求头:", request.headers)

    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    if not name or not email or not password:
        return jsonify({'error': '信息不完整'}), 400
    
    success, message = register_user(name, email, password)
    if not success:
        return jsonify({'error': message}), 400
    
    return jsonify({'email': email})

@auth_bp.route('/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if verify_user(email, password):
        return jsonify({'email': email})
    else:
        return jsonify({'error': '邮箱或密码错误'}), 400
