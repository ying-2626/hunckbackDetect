import pandas as pd
import os
import json
from utils.config import LOG_FILE
from openai import OpenAI
from datetime import datetime

def load_config():
    """Load configuration from config.json in the project root."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Traverse up to find config.json
        d = current_dir
        while os.path.dirname(d) != d:
            config_path = os.path.join(d, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            d = os.path.dirname(d)
    except Exception as e:
        print(f"Warning: Failed to load config.json: {e}")
    return {}

CONFIG = load_config()

# 配置移到类外部
OPENAI_API_KEY = CONFIG.get("DASHSCOPE_API_KEY")
if not OPENAI_API_KEY:
    # 可以在这里抛出异常，或者在调用时处理，但为了防止启动报错，这里打印警告
    # 不过用户要求严格防止泄露，且要测试模块是否可用，如果没key模块就是不可用的。
    # 既然是全局变量，抛出异常会导致导入失败。
    # 我们可以先设为 None，在 Client 初始化时会报错，或者在调用时报错。
    # OpenAI client 需要 api_key。
    print("Warning: 'DASHSCOPE_API_KEY' not found in config.json.")
    # 如果不提供 key，OpenAI client 初始化可能会失败或者默认去环境变量找。
    # 我们这里显式抛出错误或者让用户知道。
    # 为了满足“不硬编码”，我们这里就不提供默认值。
    pass

OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 创建全局的 OpenAI 客户端
try:
    OPENAI_CLIENT = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    OPENAI_CLIENT = None


class ReportGenerator:
    def generate_report(self):
        """基于视频的实时提醒：生成最近20条日志的报告"""
        try:
            df = pd.read_csv(LOG_FILE)
            logs = df.tail(20).to_string(index=False)

            completion = OPENAI_CLIENT.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一个坐姿分析医学分析者，\
                                  擅长分析驼背相关的人体节点角度比如肩-髋垂直差和头-肩垂直差并生成报告。"},
                    {"role": "user", "content": "请根据以下日志生成一份关于坐姿异常的报告,\
                                  不要使用markdown语法，\
                                  不要用头-肩垂直差等专业术语，应该用更通俗地表达使用户可以理解" + logs},
                ],
            )
            return completion.choices[0].message.content

        except Exception as e:
            return f"生成报告失败: {str(e)}"

    def generate_daily_report(self, date_str):
        """根据日期生成当天的报告"""
        try:
            # 读取日志文件
            df = pd.read_csv(LOG_FILE)

            # 转换时间戳列
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])

            # 转换输入日期并筛选
            target_date = pd.to_datetime(date_str).date()
            daily_df = df[df['Timestamp'].dt.date == target_date]

            if daily_df.empty:
                return {
                    "report": f"{date_str} 无坐姿监测数据",
                    "stats": {},
                    "date": date_str
                }


            # 生成报告内容
            logs = daily_df.to_string(index=False)

            completion = OPENAI_CLIENT.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一个坐姿分析医学分析者，\
                                              擅长分析驼背相关的人体节点角度比如肩-髋垂直差和头-肩垂直差并生成报告。"},
                    {"role": "user", "content": "请根据以下日志生成一份关于坐姿异常的报告,\
                                              不要使用markdown语法，\
                                              不要用头-肩垂直差等专业术语，应该用更通俗地表达使用户可以理解" + logs},
                ],
            )
            report = completion.choices[0].message.content

            # --- 修正 PostureStatus 字段为布尔类型 ---
            def to_bool(val):
                if isinstance(val, bool):
                    return val
                if isinstance(val, (int, float)):
                    return bool(val)
                if isinstance(val, str):
                    return val.strip().lower() == 'true'
                return False

            daily_df['PostureStatus'] = daily_df['PostureStatus'].apply(to_bool)

            # 统计正常（False）为良好坐姿
            total = ((daily_df['PostureStatus'] == False) | (daily_df['PostureStatus'] == True)).sum()
            good = (daily_df['PostureStatus'] == False).sum()
            good_posture_ratio = (good / total * 100) if total > 0 else 0

            stats = {
                "avg_ear_shoulder": daily_df['EarShoulderDiff'].mean(),
                "avg_shoulder_hip": daily_df['ShoulderHipDiff'].mean(),
                "good_posture_ratio": good_posture_ratio,
                "posture_changes": len(daily_df),
                "max_ear_shoulder": daily_df['EarShoulderDiff'].max(),
                "min_shoulder_hip": daily_df['ShoulderHipDiff'].min(),
            }

            return {
                "report": report,
                "stats": stats,
                "date": date_str
            }

        except Exception as e:
            return {
                "report": f"报告生成失败: {str(e)}",
                "stats": {},
                "date": date_str
            }
