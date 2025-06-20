import pandas as pd
from utils.config import LOG_FILE
from openai import OpenAI
from datetime import datetime

# 配置移到类外部
OPENAI_API_KEY = "sk-d43d9bf3286a40e28d86737d22aaa964"
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 创建全局的 OpenAI 客户端
OPENAI_CLIENT = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


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
                                  擅长分析驼背相关的人体节点角度比如肩-髋垂直差和头-肩垂直差并生成报告，给出改善驼背的意见。"},
                    {"role": "user", "content": "请根据以下{date_str}的坐姿日志生成一份详细的关于坐姿检测的报告,\
                                        不要使用markdown语法，但可以‘\n'换行'\
                                        当天坐姿整体评估（良好/一般/较差）主要基于posture_status而不只是hunchback_status，\
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
