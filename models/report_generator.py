import pandas as pd
from utils.config import LOG_FILE
import os
from openai import OpenAI

class ReportGenerator:
    def generate_report(self):
        # 读取CSV日志，聚合分析，生成报告
        try:
            df = pd.read_csv(LOG_FILE)
            logs=df.tail(20).to_string(index=False)  # 示例：返回最近20条日志
            client = OpenAI(
                # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
                api_key="sk-d43d9bf3286a40e28d86737d22aaa964",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一个坐姿分析医学分析者，\
                    擅长分析驼背相关的人体节点角度比如肩-髋垂直差和头-肩垂直差并生成报告。"},
                    {"role": "user", "content": "请根据以下日志生成一份关于坐姿异常的报告,\
                    不要使用markdown语法，\
                    不要用头-肩垂直差等专业术语，应该用更通俗地表达使用户可以理解" + logs},
                ],
            )
            print(completion.model_dump_json())
            return completion.choices[0].message.content

        except Exception as e:
            return f"生成报告失败: {str(e)}"

