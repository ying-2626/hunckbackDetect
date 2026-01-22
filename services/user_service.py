import json
import os
import hashlib

USERS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "users.json")

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except:
            return {}

def save_users(users):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save users: {e}")

def hash_pwd(pwd):
    return hashlib.sha256(pwd.encode('utf-8')).hexdigest()

def register_user(name, email, password):
    users = load_users()
    if email in users:
        return False, "该邮箱已注册"
    users[email] = {
        'name': name,
        'email': email,
        'password': hash_pwd(password)
    }
    save_users(users)
    return True, None

def verify_user(email, password):
    users = load_users()
    user = users.get(email)
    if not user or user['password'] != hash_pwd(password):
        return False
    return True
