import pandas as pd
from utils.config import LOG_FILE

class ReportGenerator:
    def generate_report(self):
        # 读取CSV日志，聚合分析，生成报告
        try:
            df = pd.read_csv(LOG_FILE)
            # ...可以在这里添加聚合/分析逻辑...
            return df.tail(10).to_string(index=False)  # 示例：返回最近10条日志
        except Exception as e:
            return f"生成报告失败: {str(e)}"
