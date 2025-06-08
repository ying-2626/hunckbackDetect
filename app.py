from flask import Flask, jsonify, request
from flask_mail import Mail
from core.scheduler import Scheduler
from utils.config import SERVER_CONFIG

app = Flask(__name__)
mail = Mail(app)
s = Scheduler(mail,app)  # 传入app实例

@app.route('/')
def index():
    return """
    <h1>姿势监测系统</h1>
    <p>发送图片进行分析</p>
    <form action="/analyze" method="post" enctype="multipart/form-data">
        <input type="file" name="image">
        <input type="submit" value="分析姿势">
    </form>
    """

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status':'ok'})

# start background scheduler on app start
def start_background_tasks():
    s.start()

if __name__ == '__main__':
    start_background_tasks()
    app.run(host='0.0.0.0', port=SERVER_CONFIG['PORT'])